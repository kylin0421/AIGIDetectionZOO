import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score



from .networks import Net as RPTC
from .networks import initWeights
from ..api import AIGIDetection_API

def ED(img):
    r1,r2 = img[:,0:-1,:], img[:,1::,:]
    r3,r4 = img[:,:,0:-1], img[:,:,1::]
    r5,r6 = img[:,0:-1,0:-1], img[:,1::,1::]
    r7,r8 = img[:,0:-1,1::], img[:,1::,0:-1]
    s1 = torch.sum(torch.abs(r1 - r2)).item()
    s2 = torch.sum(torch.abs(r3 - r4)).item()
    s3 = torch.sum(torch.abs(r5 - r6)).item()
    s4 = torch.sum(torch.abs(r7 - r8)).item() 
    return s1+s2+s3+s4


def processing_RPTC(img,seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    rate = 3
        
    num_block = int(pow(2,rate))
    patchsize = int(256/num_block)
    randomcrop = torchvision.transforms.RandomCrop(patchsize)
    
    minsize = min(img.size)
    if minsize < patchsize:
        img = torchvision.transforms.Resize((patchsize,patchsize))(img)
    
    img = torchvision.transforms.ToTensor()(img)
    # img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    imgori = img.clone().unsqueeze(0)
    img_template = torch.zeros(3, 256, 256)
    img_crops = []
    for i in range(num_block*num_block*3):
        cropped_img = randomcrop(img)
        texture_rich = ED(cropped_img)
        img_crops.append([cropped_img,texture_rich])

    img_crops = sorted(img_crops,key=lambda x:x[1])

    count = 0
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:,ii*patchsize:(ii+1)*patchsize,jj*patchsize:(jj+1)*patchsize] = img_crops[count][0]
            count += 1
    img_poor = img_template.clone().unsqueeze(0)

    count = -1
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:,ii*patchsize:(ii+1)*patchsize,jj*patchsize:(jj+1)*patchsize] = img_crops[count][0]
            count -= 1
    img_rich = img_template.clone().unsqueeze(0)
    img = torch.cat((img_poor,img_rich),0)
    
    return img

def processing_RPTC_with_random(img):

    rate = 3
        
    num_block = int(pow(2,rate))
    patchsize = int(256/num_block)
    randomcrop = torchvision.transforms.RandomCrop(patchsize)
    
    minsize = min(img.size)
    if minsize < patchsize:
        img = torchvision.transforms.Resize((patchsize,patchsize))(img)
    
    img = torchvision.transforms.ToTensor()(img)
    # img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    imgori = img.clone().unsqueeze(0)
    img_template = torch.zeros(3, 256, 256)
    img_crops = []
    for i in range(num_block*num_block*3):
        cropped_img = randomcrop(img)
        texture_rich = ED(cropped_img)
        img_crops.append([cropped_img,texture_rich])

    img_crops = sorted(img_crops,key=lambda x:x[1])

    count = 0
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:,ii*patchsize:(ii+1)*patchsize,jj*patchsize:(jj+1)*patchsize] = img_crops[count][0]
            count += 1
    img_poor = img_template.clone().unsqueeze(0)

    count = -1
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:,ii*patchsize:(ii+1)*patchsize,jj*patchsize:(jj+1)*patchsize] = img_crops[count][0]
            count -= 1
    img_rich = img_template.clone().unsqueeze(0)
    img = torch.cat((img_poor,img_rich),0)
    
    return img.unsqueeze(0)





CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
class PATCHCRAFT_API(AIGIDetection_API):
    def __init__(self,
                model_path=os.path.join(CURRENT_DIR, 'weights', 'RPTC.pth'),           # Currently, only official weights supported
                use_cuda=True):
        super().__init__(preprocess=processing_RPTC_with_random)
        
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = RPTC()
        self.model.apply(initWeights)
        state_dict = torch.load(model_path, map_location='cpu')
        netC_state_dict = state_dict['netC']
        

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in netC_state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v

        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()


    def get_preprocess(self):
        return transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])


    # Detections
    # All detection return a list of normalized logits where closer to 1 indicates likely being AI-generated and 0 indicates likely being real.
    # Only detect_image will do preprocessing using the self.preprocess parameter. 
    def detect_image(self, image: np.ndarray) -> float:
        self.model.eval()

        with torch.no_grad():
            input_tensor = self.preprocess(image).to(self.device)
            score = self.model(input_tensor).sigmoid().item()
            return score  



    def detect_batch(self, batched_img: torch.Tensor) -> torch.Tensor:
        """
        对每张图片构造 [2, 3, 256, 256] rich/poor 对，然后堆叠成 [B, 2, 3, 256, 256] 喂给模型。
        Args:
            batched_img: (B, 3, H, W)
        Returns:
            (B,) real probability
        """
        batch_size = batched_img.size(0)
        img_pairs = []

        for i in range(batch_size):
            img = transforms.ToPILImage()(batched_img[i].cpu())  # (3,H,W) → PIL
            pair = processing_RPTC_with_random(img)              # shape [2, 3, 256, 256]
            img_pairs.append(pair)                               # 不要 unsqueeze(0)

        x = torch.stack(img_pairs, dim=0).to(self.device)  # shape [B, 2, 3, 256, 256]

        with torch.no_grad():
            logits = self.model(x)  # shape: [B, 1] or [B]
            probs = torch.sigmoid(logits).squeeze(dim=-1)
            return probs  # 假设输出的是 fake prob
        

    # Expects a DataLoader which already contains preprocessed images
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
        


    def train(self):
        raise NotImplementedError("Training not implemented in current version, " \
        "please try to update aigidetection_zoo or contact the author via GitHub: https://github.com/kylin0421/AIGIDetectionZOO for support.")






