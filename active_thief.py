import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
# from scipy.stats import entropy


def entropy(p):
    return -torch.sum(p * torch.log10(p))


def least_confidence(p):
    return torch.max(p)


def margin_sampling(p):
    sample = torch.topk(p, k=2)
    return sample[0][0] - sample[0][1]


class ActiveThief(object):

    def __init__(self, data, batch_size=50):
        self.batch_size = batch_size

        self.indices = []
        self.dataset = data
        self.prev_agreement = 1
        self.prev_s_prob = None
        self.prev_samples = None
        self.s_prob = None
        self.samples = []
        self.labels = []

    def cluster_seeding(self, s_prob, batch_size):
        print("Finding Clusters")
        kmeans = KMeans(n_clusters=10, max_iter=300)
        kmeans.fit(s_prob)

        distances = []
        for i, train in enumerate(s_prob, 0):
            if i % 5000 == 0:
                print("Calculating distances:", i)
            distance = 0
            for center in kmeans.cluster_centers_:
                distance += np.linalg.norm(center - train.numpy())
            distances.append(distance)
        sample_index = np.flip(np.argsort(distances))[:batch_size]
        for i in sample_index:
            self.indices.append(i)

        samples = Subset(self.dataset, sample_index)
        self.samples = sample_index
        return DataLoader(samples, num_workers=8)

    def random_seeding(self, batch_size):
        sample_index = np.random.choice(
            np.setdiff1d(np.arange(self.dataset.data.shape[0]), self.indices), size=batch_size, replace=False)
        for index in sample_index:
            self.indices.append(index)
        samples = Subset(self.dataset, sample_index)
        self.samples = sample_index
        return DataLoader(samples, num_workers=8)

    def subset_selection(self, s_prob, strategy="entropy", labels=None):
        self.s_prob = s_prob
        if strategy is "random":
            return self.random_seeding(self.batch_size)
        if strategy is "entropy":
            return self.entropy_selection(s_prob)
        elif strategy is "lc":
            return self.lc_selection(s_prob)
        elif strategy is "margin_sampling":
            return self.margin_sampling_selection(s_prob)
        elif strategy is "cluster":
            return self.cluster_sampling(s_prob, labels)
        elif strategy is "cluster_entropy":
            return self.cluster_uncertainty(s_prob, labels)
        elif strategy is "uniform":
            return self.equal_labels(s_prob)
        elif strategy is "uniform_entropy":
            return self.equal_entropy(s_prob)
        elif strategy is "cluster_uniform_entropy":
            return self.batch_uncertainty_uniform(s_prob, labels)
        else:
            return None

    def entropy_selection(self, probabilities):
        confidence = []
        for p in probabilities:
            confidence.append(entropy(p))

        batch_indices = []
        for i in np.flip(np.argsort(confidence)):
            if len(batch_indices) == self.batch_size:
                break
            if i not in self.indices:
                batch_indices.append(i)
                self.indices.append(i)
        samples = Subset(self.dataset, batch_indices)
        self.samples = batch_indices
        return DataLoader(samples, num_workers=8)

    def lc_selection(self, probabilities):
        confidence = []
        for p in probabilities:
            confidence.append(least_confidence(p))

        batch_indices = []
        for i in np.argsort(confidence):
            if len(batch_indices) == self.batch_size:
                break
            if i not in self.indices:
                batch_indices.append(i)
                self.indices.append(i)
        samples = Subset(self.dataset, batch_indices)
        self.samples = batch_indices
        return DataLoader(samples, num_workers=8)

    def margin_sampling_selection(self, probabilities):
        confidence = []
        for p in probabilities:
            confidence.append(margin_sampling(p))

        batch_indices = []
        for i in np.argsort(confidence):
            if len(batch_indices) == self.batch_size:
                break
            if i not in self.indices:
                batch_indices.append(i)
                self.indices.append(i)
        samples = Subset(self.dataset, batch_indices)
        self.samples = batch_indices
        return DataLoader(samples, num_workers=8)

    def cluster_sampling(self, expected, labels):
        distances = []
        for label in labels:
            self.labels.append(label.numpy()[0])

        kmeans = KMeans(n_clusters=10, max_iter=300)
        kmeans.fit(self.labels)

        for i, train in enumerate(expected, 0):
            if i % 5000 == 0:
                print("Calculating distances:", i)
            distance = 0
            if i not in self.indices:
                for center in kmeans.cluster_centers_:
                    distance += np.linalg.norm(center - train.numpy())
            distances.append(distance)

        batch_indices = []
        for i in np.flip(np.argsort(distances)):
            if len(batch_indices) == self.batch_size:
                break
            if i not in self.indices:
                batch_indices.append(i)
                self.indices.append(i)
        samples = Subset(self.dataset, batch_indices)
        self.samples = batch_indices
        return DataLoader(samples, num_workers=8)

    def cluster_uncertainty(self, expected, labels):
        for label in labels:
            self.labels.append(label.numpy()[0])

        confidence = []
        for i, train in enumerate(expected, 0):
            confidence.append(entropy(train))

        kmeans = KMeans(n_clusters=10, max_iter=300)
        kmeans.fit(self.labels)

        confidence_index = np.flip(np.argsort(confidence))

        counter = 0
        distances = []
        distances_index = []
        for i, train in enumerate(confidence_index, 0):
            if counter >= 10000:
                break
            if i % 5000 == 0:
                print("Calculating distances:", i)
            distance = 0
            if i not in self.indices:
                counter += 1
                for center in kmeans.cluster_centers_:
                    distance += np.linalg.norm(center - expected[train].numpy())
            distances_index.append(i)
            distances.append(distance)

        batch_indices = []
        for j in np.flip(np.argsort(distances)):
            i = confidence_index[distances_index[j]]
            if len(batch_indices) == self.batch_size:
                break
            if i not in self.indices:
                batch_indices.append(i)
                self.indices.append(i)

        samples = Subset(self.dataset, batch_indices)
        self.samples = batch_indices
        return DataLoader(samples, num_workers=8)

    def equal_labels(self, expected):
        batch_indices = []
        _, predicted = torch.max(expected, 1)
        for i in range(10):
            for j in range(predicted.shape[0]):
                if len(batch_indices) == (i+1) * 100:
                    break
                if predicted[j] == i and j not in self.indices:
                    batch_indices.append(j)
                    self.indices.append(j)

        if len(batch_indices) < self.batch_size:
            sample_index = np.random.choice(
                np.setdiff1d(np.arange(self.dataset.data.shape[0]), self.indices),
                size=self.batch_size - len(batch_indices), replace=False)
            for i in sample_index:
                self.indices.append(i)
                batch_indices.append(i)

        samples = Subset(self.dataset, batch_indices)
        self.samples = batch_indices
        return DataLoader(samples, num_workers=8)

    def equal_entropy(self, probabilities):
        confidence = []
        for p in probabilities:
            confidence.append(entropy(p))
        confidence = np.flip(np.argsort(confidence))

        batch_indices = []
        _, predicted = torch.max(probabilities, 1)
        for i in range(10):
            for j in confidence:
                if len(batch_indices) == (i + 1) * 100:
                    break
                if predicted[j] == i and j not in self.indices:
                    batch_indices.append(j)
                    self.indices.append(j)

        for i in confidence:
            if len(batch_indices) >= self.batch_size:
                break
            if i not in self.indices:
                self.indices.append(i)
                batch_indices.append(i)


        samples = Subset(self.dataset, batch_indices)
        self.samples = batch_indices
        return DataLoader(samples, num_workers=8)

    def batch_uncertainty_uniform(self, expected, labels):
        for label in labels:
            self.labels.append(label.numpy()[0])

        confidence = []
        for i, train in enumerate(expected, 0):
            confidence.append(entropy(train))

        kmeans = KMeans(n_clusters=10, max_iter=300)
        kmeans.fit(self.labels)

        confidence_index = np.flip(np.argsort(confidence))

        counter = 0
        distances = []
        distances_index = []
        for i, train in enumerate(confidence_index, 0):
            if counter >= 20000:
                break
            if i % 5000 == 0:
                print("Calculating distances:", i)
            distance = 0
            if i not in self.indices:
                counter += 1
                for center in kmeans.cluster_centers_:
                    distance += np.linalg.norm(center - expected[train].numpy())
            distances_index.append(i)
            distances.append(distance)

        batch_indices = []
        for i in range(10):
            for j in np.flip(np.argsort(distances)):
                x = confidence_index[distances_index[j]]
                if len(batch_indices) == (i + 1) * 100:
                    break
                if x not in self.indices:
                    batch_indices.append(x)
                    self.indices.append(x)

        samples = Subset(self.dataset, batch_indices)
        self.samples = batch_indices
        return DataLoader(samples, num_workers=8)

    def stopping_criteria(self, a):
        if self.prev_samples is None:
            self.prev_samples = self.samples
            self.prev_s_prob = self.s_prob
            return False

        current_entropy = 0
        for i in self.samples:
            current_entropy += entropy(self.s_prob[i])

        prev_entropy = 0
        for i in self.prev_samples:
            prev_entropy += entropy(self.prev_s_prob[i])

        current_entropy = current_entropy / len(self.samples)
        prev_entropy = prev_entropy / len(self.prev_samples)

        self.prev_samples = self.samples
        self.prev_s_prob = self.s_prob
        compare = abs(prev_entropy - current_entropy)
        print("Current agreement is:", compare)
        return compare < a

