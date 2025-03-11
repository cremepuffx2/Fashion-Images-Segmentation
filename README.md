# Fashion-Images-Segmentation
![image](https://github.com/user-attachments/assets/35c38e57-8a70-4008-ac85-efcfa80d03a2)
<p align="center"><i>Figure 1: Current Swapaholic UI</i></p>

With the ongoing AI wave, the current merchant sphere is evolving rapidly. Hence, businesses need to stay competitive and respond to the rapidly changing market conditions. Swapaholic, one of the biggest platforms in Singapore for buying and selling clothes between consumers, has root problems with cumbersome data consolidation processes and an inefficient online marketplace. The main problem of the platform is that it does not use unified data from multiple sources, limiting its ability to leverage information efficiently. As a result, the company is unable to respond in time to consumer demands and get insights from their behaviour, which in turn leads to a lower competitiveness and attractiveness to customers.

An article from “Scaling Up Excellence” highlights the pervasive data fragmentation as a prominent cause for issues that retail businesses face. It states that the data should be unified so the business can update information about a product without any delays and inaccuracies. (Morsen, 2023).

Further compounding these issues is the labour-intensive and inefficient seller onboarding and listing process. Due to cumbersome factors such as time-consuming manual inputs for listing, the user experience for sellers is likely to be unpleasant. It takes around 10 minutes to create one card for an item which increases the average session duration on the website. However, the funnel abandonment rate also increases in tandem, leading to lower consumption of the website in the future. This issue degrades user experience and decreases the operational effectiveness of the marketplace. Hence, the implementation of CV technologies has the potential to overcome all the problems and lower the number of labour units that need to operate the system.

The project aims to discover the best-performing ML model among the ones highlighted above, to integrate into Swapaholic’s user interface using CV techniques and automate the apparel listing process on its website. This will help to reduce, and eventually avoid, the manual effort required from the seller to manually enter details of his or her listing, and the accuracy of the details of each listing is expected to improve as well. Overall, we are not only aiming to boost operational efficiency within Swapaholic, but we are also looking to increase their user retention and user satisfaction metrics, generating growth in Swapaholic’s overall user sales.

One such example of a successful implementation of process automation in the retail sector using CV can be observed in Amazon Go. The company was a pioneer in this field, where they utilised CV and deep learning algorithms to decrease the amount of theft, track customers, and obtain data on their customers’ shopping behaviour habits while decreasing management costs as well (Honaman, 2024). A CV implementation in monotonous and complex human tasks is key to handling many problems at once. Taking inspiration from Amazon Go, the project hopes that integration of CV techniques into Swapaholic’s operations will contribute to a faster and more efficient way of product categorization.

### Dataset
In our research, we used the dataset “iMaterialist (Fashion) 2020 at FGVC7” from the Kaggle website. This dataset consists of 48,826 files that contain files related to clothing and accessories. Segmentation and classification tasks are performed using the labels each image has, and these labels are stored in the value ImageId. We use the 2020 version as the dataset was expanded and attributes were correctly relabeled since the 2019 version.

The dataset consists of images of people wearing clothes in different poses, and it also contains the masks and annotations for them. The masks are used to make accurate segmentation between images.

The dataset is 23.7 GB in size and includes:
- Images folder
- JSON file containing category and attribute label descriptions
- CSV file containing annotations

The CSV file has 4 columns:
- “ImageId” is a label for an image
- “EncodedPixels” are run-length encoded masks with pixel-wise annotations
- “ClassId” is an ID for each mask which represents the category
- “AttributesIds” are the IDs

### DETR-ResNet-50 Model Architecture

#### Description
![image](https://github.com/user-attachments/assets/32e0d2bf-fc1f-4ea7-8de8-9344b7e26afa)
<p align="center"><i>Figure 2: DETR Architecture</i></p>

DETR-ResNet-50 (Detection Transformer) is an end-to-end object detection model that uses a transformer-based architecture to directly predict object categories and bounding boxes without requiring region proposals. It leverages a deep ResNet-50 backbone to extract image features and a transformer-based decoder to perform attention-based predictions.

### Data Pre-processing
The data preprocessing pipeline implements a FashionpediaDataset class, featuring two key components: an ID mapping system and a dataset processor. The IDMapper class handles the conversion of arbitrary category and attribute IDs to consecutive integers, maintaining bidirectional mappings that can be saved and loaded for consistency across sessions. The FashionpediaDataset class inherits from PyTorch's Dataset class and processes image data alongside their corresponding bounding boxes, category labels, and attribute annotations. Images are resized to 384x384 pixels and normalized using ImageNet statistics. The dataset is split into 70% train, 15% validation, and 15% test. A custom collation function is implemented for batch processing, outputting a dictionary containing tensor batches of pixel values, bounding box coordinates, category labels, and attribute labels, ready for model training.

### Loss Functions
The loss functions implementation presents a multi-task learning framework that employs uncertainty-based task weighting for simultaneous multi-class category classification, multi-label attribute prediction, and instance segmentation. The architecture utilizes learnable uncertainty parameters through the UncertaintyWeights module, which dynamically balances the relative importance of each task by learning log variances that are converted to precision weights. The loss computation incorporates Hungarian matching to optimize:

- **Jaccard Loss (1 - mIoU) for mask predictions**
![image](https://github.com/user-attachments/assets/24a8a203-f7ee-46e1-b81e-bcc4e8aed6de)
- **Cross-entropy loss for category predictions**
- **Binary cross-entropy loss for attribute set predictions**

The Hungarian matching algorithm is employed to optimise the assignment between predictions and ground truth values, minimising the total loss:
![image](https://github.com/user-attachments/assets/645e9724-3fb3-4ef7-bb38-6de8cd64a2b7)
The final loss function combines these task-specific losses using the learned uncertainty weights, with an added regularisation term to prevent the model from trivially minimising losses by increasing uncertainties. This approach allows the model to automatically adapt the contribution of each task to the total loss based on their relative difficulties and uncertainties, potentially leading to more balanced and effective multi-task learning.

### Custom DETR-ResNet-50 Model for Fashionpedia
We implemented a custom extension of DETR-ResNet-50 for the Fashionpedia dataset by modifying the architecture to handle instance segmentation, category classification, and multi-label attribute prediction. The model builds upon the pre-trained DETR-ResNet-50 backbone (frozen during training) from the Hugging Face Hub and adds two task-specific linear classifier heads for category and attribute set prediction. The architecture maintains DETR's original segmentation capabilities while extending its functionality to handle multi-class category classification and multi-label attribute classification through parallel classification heads.

### Training
![image](https://github.com/user-attachments/assets/a7ad60f3-676c-44d3-a74b-4172e8914f78)
<p align="center"><i>Figure 3: Training vs Validation Loss</i></p>

The training pipeline implements the custom DETR-ResNet-50 model that leverages the Hungarian algorithm for optimal bipartite matching between predicted and ground truth masks, ensuring efficient assignment of segment predictions to target labels. Using an AdamW optimizer with a conservative learning rate of 1e-5 for stable convergence, the pipeline incorporates checkpointing and memory optimization features.

### Evaluation
The evaluation pipeline loads the best-performing model checkpoint based on validation loss and evaluates it on test data in batches to manage memory efficiently. For each batch, it computes three key metrics:
- **Mean Intersection over Union (mIoU) for mask predictions**
- **Dice coefficient for category predictions**
- **Dice coefficient for attribute set predictions**

### Visualisation
![image](https://github.com/user-attachments/assets/aef9ab5a-8282-4c11-a52c-60f3a50cc112)
<p align="center"><i>Figure 4: Sample Segmentation for DETR-Resnet-50</i></p>

![image](https://github.com/user-attachments/assets/70dab952-3fd5-4e6b-b299-9989fa3ee094)
<p align="center"><i>Figure 5: Another Sample Segmentation for DETR-Resnet-50</i></p>
The visualization pipeline was designed to process images through the CustomDETRForFashionpedia model, incorporating both instance segmentation, category classification, and attribute set prediction. The system generates mask predictions with confidence scores, and the pipeline was structured to refine these masks using non-maximum suppression (NMS) with an IoU threshold of 0.5 to eliminate overlapping detections. The planned approach involved selecting the top-k predictions (initially 100, filtered to 15 after NMS) and interpolating the mask logits to match the original image dimensions.
<p></p></p>
However, in practice, the model struggled to produce accurate visual outputs, likely due to challenges in learning mask boundaries effectively. The predicted masks were either too noisy, inconsistent, or failed to align correctly with objects in the images. As a result, while the pipeline was built to overlay masks with corresponding category labels and attributes, the final outputs were not reliable for meaningful interpretation. Future work could explore fine-tuning the model further, incorporating additional post-processing steps, or experimenting with alternative architectures to improve visualization quality.

### A/B Testing
For our user testing, we employed Maze.co and two Figma mockups to assess the usability and user preference of listings UI: https://app.maze.co/report/Listing-AB/b2nx547m2gm3q19/intro

The original manual Listing A
The newly proposed automated Listing B
Participants were tasked to successfully navigate the uploading of a clothing image for listing twice; once using Listing A and once using Listing B. Out of the initial 52 participants, 25 (48%) fully completed all tasks. Outliers—such as those who spent over 685.84 seconds on the second listing or skipped tasks—were excluded from the analysis to focus on genuine user interaction data. The findings showed a slight preference for the automated Listing B over the original Listing A.

![image](https://github.com/user-attachments/assets/64c9971d-0913-4242-87f3-24360a28d303)
<p align="center"><i>Figure 6: Listing Method Survery Results</i></p>

Using a 5-point scale of preference (where Listing A = 1 and Listing B = 5), we calculated a mean preference of 3.81, with both mode and median at 4. This shows a generally favourable response to Listing B. The automated Listing B also demonstrated efficiency gains, with a lower average completion time of 64 seconds compared to Listing A’s 109 seconds. Listing B also had a narrower spread in task completion times (standard deviation of 19.96 seconds versus Listing A's 33.54 seconds), suggesting a more consistent user experience.

Taken together, these results highlight several potential benefits of automated features in Listing B. The automation appears to streamline the process, reducing both completion time and variability, making it more convenient and user-friendly. The automated features in Listing B likely reduce cognitive load by minimising manual steps and decision points, allowing users to complete listings faster and with less effort. The lower standard deviation for Listing B’s completion times can also be attributed to the automated steps helping users encounter fewer unexpected interactions, resulting in smoother, more consistent experiences across participants.
