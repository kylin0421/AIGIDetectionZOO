import warnings
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import torch




class AIGIDetection_API():
    def __init__(self,preprocess=None):
        assert (preprocess is None) or callable(preprocess), "preprocess must be a callable function that takes in a PIL image and returns a tensor."
        self.preprocess = preprocess
        if self.preprocess is None:
            warnings.warn("No preprocessing function provided. This is expected only when you are only using 'detect_dataloader()' and passing in a dataloader with preprocessing.", UserWarning)



    # Detections
    # All detection return a list of normalized logits where closer to 1 indicates likely being AI-generated and 0 indicates likely being real.
    # Except detect_dataloader(), all detection will do preprocessing using the self.preprocess parameter. 
    def detect_image(self, image):
        if self.preprocess is not None:
            image = self.preprocess(image)
        pass


    def detect_batch(self, images):
        if self.preprocess is not None:
            images = [self.preprocess(image) for image in images]
        pass

    # detect all images in a dataloader
    # input must be torch.utils.data.DataLoader which contains (image,label) in each sample, label is used only when return_metrics is True
    # label == True indicates AI-generated
    def detect_dataloader(self, 
                        dataloader,
                        return_metrics=False,
                        metrics=['accuracy','f1','recall','precision','auroc']):
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            probs = self.detect_batch(images)    # 已经是 [0,1] 概率
            preds = (probs > 0.5).long()         # 二分类阈值化

            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_probs.append(probs.detach().cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        if return_metrics:
            results = {}
            if 'accuracy' in metrics:
                results['accuracy'] = accuracy_score(all_labels, all_preds)
            if 'f1' in metrics:
                results['f1'] = f1_score(all_labels, all_preds)
            if 'recall' in metrics:
                results['recall'] = recall_score(all_labels, all_preds)
            if 'precision' in metrics:
                results['precision'] = precision_score(all_labels, all_preds)
            if 'auroc' in metrics:
                results['auroc'] = roc_auc_score(all_labels, all_probs)
            return results,all_preds,all_probs
        else:
            return all_preds, all_probs






    # training
    # currently no training supported!
    def train(self, dataloader, optimizer, criterion, num_epochs=1):
        self.model.train()
        for epoch in range(num_epochs):
            for images, labels in dataloader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        pass