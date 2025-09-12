import torch



from torchvision import transforms
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from huggingface_hub import hf_hub_download
import cv2
from PIL import Image
from io import BytesIO
import warnings
import re

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=re.escape("Attempting to run cuBLAS, but there was no current CUDA context!")
)


from .LGrad_models import build_model
from ..api import AIGIDetection_API
from .networks.resnet import resnet50

GEN_MODEL_PREPROCESS =  transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
#            transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
        ])


MODEL_PREPROCESS = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
                ])


def normlize_np(img):
    img -= img.min()
    if img.max()!=0: img /= img.max()
    return img * 255.   


class LGRAD_API(AIGIDetection_API):
    def __init__(self, 
                gen_model_preprocess=GEN_MODEL_PREPROCESS,
                model_preprocess=MODEL_PREPROCESS,
                official_weights=True,
                use_cpu=False,
                ):
        super().__init__(preprocess=model_preprocess)  # we don't actually use self.preprocess since there are two different preprocess pipelines
        self.gen_preprocess=gen_model_preprocess
        self.model_preprocess=model_preprocess


        self.get_weights(official=official_weights)  # this sets self.weights
        self.use_cpu=use_cpu
        self.device = torch.device('cpu' if use_cpu else 'cuda')


        model = resnet50(num_classes=1)
        state_dict = torch.load(self.classifier_weights, map_location='cpu')
        model.load_state_dict(state_dict['model'],strict=True)
        model.to(self.device)

        gen_model = build_model(gan_type='stylegan',
            module='discriminator',
            resolution=256,
            label_size=0,
            # minibatch_std_group_size = 1,
            image_channels=3)
        gen_model.load_state_dict(torch.load(self.process_weights), strict=True)
        if(not self.use_cpu):
            gen_model.cuda()
        gen_model.eval()

        self.model = model
        self.gen_model=gen_model




    def get_preprocess(self):
        #LGrad model has two preprocessing, we only return the model_preprocess here since the gen_model_preprocess is more of a built-in mechanism.
        return self.model_preprocess




    def get_weights(self,
                    official=True):    

        weights_dir = "aigidetection_zoo/LGrad/weights"
        os.makedirs(weights_dir, exist_ok=True)


        classifier_path = os.path.join(weights_dir, "LGrad.pth")

        if not os.path.exists(classifier_path):
            print(f"{classifier_path} not found, downloading...")
            self.classifier_weights = hf_hub_download(
                repo_id="slxhere/LGrad",
                filename="LGrad.pth",
                local_dir=weights_dir
            )
            print(f"Downloaded weights to {self.classifier_weights}")
        else:
            self.classifier_weights = classifier_path   # 关键补充
            print(f"Found existing weights at {classifier_path}, skip downloading.")


        process_path = os.path.join(weights_dir, "karras2019stylegan-bedrooms-256x256_discriminator.pth")

        if not os.path.exists(process_path):
            print(f"{process_path} not found, downloading...")
            self.process_weights = hf_hub_download(
                repo_id="slxhere/LGrad",
                filename="karras2019stylegan-bedrooms-256x256_discriminator.pth",
                local_dir=weights_dir
            )
            print(f"Downloaded weights to {self.process_weights}")
        else:
            self.process_weights = process_path   # 关键补充
            print(f"Found existing weights at {process_path}, skip downloading.")



    # Detections
    # All detection return a list of normalized logits where closer to 1 indicates likely being AI-generated and 0 indicates likely being real.
    # Only detect_image will do preprocessing using the self.preprocess parameter. 
    def detect_image(self, img):

        img_list = []
        img_list.append(torch.unsqueeze(self.gen_preprocess(img),0))
        img=torch.cat(img_list,0)
        img_cuda = img.to(torch.float32)
        img_cuda= img_cuda.to(self.device)
        img_cuda.requires_grad = True
        pre = self.gen_model(img_cuda)
        self.gen_model.zero_grad()
        grads = torch.autograd.grad(pre.sum(), img_cuda, create_graph=True, retain_graph=True, allow_unused=False)[0]
        for idx,grad in enumerate(grads):
            img_grad = normlize_np(grad.permute(1,2,0).cpu().detach().numpy())
        img_grad = img_grad.astype("uint8")
        retval, buffer = cv2.imencode(".png", img_grad)
        if retval:
            img = Image.open(BytesIO(buffer)).convert('RGB')
        else:
            print("Failed to save to memory")
    
        img = self.model_preprocess(img)



        with torch.no_grad():
            in_tens = img.unsqueeze(0)
            if(not self.use_cpu):
                in_tens = in_tens.cuda()
            prob = self.model(in_tens).sigmoid().item()

        return prob
    

    # Input must be of dims (B,C,H,W)
    # Returns Tensor (B,)
    def detect_batch(self, imgs):
        # 保证 float32 并移动到 device
        imgs = imgs.to(torch.float32).to(self.device)
        imgs.requires_grad = True

        # 1. 先跑生成器
        pre = self.gen_model(imgs)
        self.gen_model.zero_grad()

        # 2. 计算梯度
        grads = torch.autograd.grad(
            pre.sum(), imgs,
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]

        # 3. 可选：对 batch 中的每张图做可视化
        img_grads = []
        for b in range(grads.shape[0]):
            grad = grads[b]  # shape (C,H,W)
            img_grad = normlize_np(
                grad.permute(1, 2, 0).cpu().detach().numpy()
            )
            img_grad = img_grad.astype("uint8")
            retval, buffer = cv2.imencode(".png", img_grad)
            if retval:
                pil_img = Image.open(BytesIO(buffer)).convert('RGB')
                img_grads.append(pil_img)
            else:
                print("Failed to save to memory")

        # 4. 拼 batch
        if len(img_grads) > 0:
            in_tens = torch.stack(img_grads, dim=0)
        else:
            raise RuntimeError("No valid grads in detect_batch")

        if not self.use_cpu:
            in_tens = in_tens.cuda()

        # 5. 得到概率 (B,)
        with torch.no_grad():
            prob = self.model(in_tens).sigmoid().squeeze()

        return prob





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
        "please try to update aigidetection_zoo "
        "or contact the author via GitHub: https://github.com/kylin0421/AIGIDetectionZOO for support.")