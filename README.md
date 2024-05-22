# CV task
### the final model is deployed at https://huggingface.co/spaces/saatvik1879/CVtask do check it out 😁
### lucidchart of model workings - https://lucid.app/lucidchart/9055093a-1186-4031-be61-4183f02982d5/edit?viewport_loc=85%2C-39%2C1579%2C1016%2C0_0&invitationId=inv_9ab7f7d6-54c1-452c-b8ae-0f5a955f69ff
## Acquiring the datasets
My first thought was to train a model locally for this task. but, I couldn't get Pytorch running locally. Then I thought of running it on colab but the number of images in the file is too large and It will take quite a lot of time to train my model. So, I finally concluded using openCV and Kaggle. Then that didn't work so I had to train a CV model on my Macbook without GPU 

(my first idea was that I would use OpenCV to extract features and then use random forest classifier on the extracted features to perform classification but that didn't work quite well )



the images used to train my model will be a subset of this dataset(more specifically I used only the test folder of this dataset because that is all my Mac could handle )-
https://www.kaggle.com/datasets/ashishjangra27/gender-recognition-200k-images-celeba




So, to test the performance of the model I will use a small subset of the dataset - https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

## building model

### Attempt 1: using dlib (unsuccessful one)

#### This whole attempt was a mess and I finally scrapped it to read the details of it read below else move to attempt 2

first I shall preprocess the images  then the model takes preprocessed images and performs feature extraction on them. the features extracted are the ones which are not covered by a mask and then are stored in a feature vector. then the feature vector is passed to the random forest classifier to perform classification.

#### importing necessary libraries
importing libraries like  cv2, randomforestclassifier etc.
#### preprocessing the image
First, It will find where the faces are located and crop the image such that the face is centred using dlib.

and then it is resized to 200x200 and converted into greyscale.

#### extracting feature vectors from preprocessed images 
I am using openCV to extract features like hairstyle, facial geometry, skin tone and geometry features.

first I am using dlib's shape predictor to find the locations of landmarks on the face.


the features extracted here are `hair texture`, `skin tone` and `skin texture`.

#### hair texture
first, the region containing hair has been extracted i.e. from landmarks extracted from the landmark_detector of cv2

then cv2.laplacian() is applied which is an edge detection algorithm. It captures texture and variations in hair

.var() operator then calculates the spread of pixels higher variance indicates textured hairstyle whereas low variance indicates smoother texture.

#### skin tone
for skin tone, I am first cropping out the forehead region and then taking out the mean of the pixel values to get the skin tone

#### skin texture
I am taking the forehead region cropped out from above and applying Laplacian on it. 

#### random forests 
Lastly, I use random forests to process the feature vector. and classify it according to the gender

#### problems faced:
first I couldn't get the main library dlib to run
so I tried openCV's alternatives but that didn't work out either as I couldn't load the required models from opencv too so I skipped the part where dlib was required and just used some simple opencv algorithms but due to that the accuracy (=0.56)of the model suffered quite a bit. so now we're back to the drawing board and now comes attempt 2.
### Attempt 2: using transfer learning(successful one)
the working mechanism is the final model is that first it takes the image finds faces in the images and passes the found images to the model and the model then applies convolutions on the image to finally pass it through a sigmoid to determine the gender
TH model has been deployed on https://huggingface.co/spaces/saatvik1879/CVtask(Please DM me if the space is private I will turn it into public if I forget)
#### importing the necessary libraries
import libraries like opencv, os, tf, numpy, etc. the VGG model and dense, pooling layers 
#### preprocessing function
first, the image is read from its path. and then the grayscale version of the image is stored in a variable, this variable is then passed to OpenCV's cascade classifier which finds the coordinates at which faces are present. and the image is cropped such that this face is at the centre of the image. the lower half of the face's pixels are converted into black pixels. then this is passed tf.keras.applications.vgg16's preprocess function. If no face is found (It is happening in the case of the image being already cropped) I am passing using the whole image at the step where I cropped only the face.
#### Preparing the data
to prepare the data I am taking all the images and passing them through the preprocessing function and appended to the list named data

P.S. I dared to save and load this data and upload this data onto Colab and use its GPU to train my model and use it there which first crashed Colab then my computer crashed as the size of the data is 11 GB

#### test train split and using data generators
I am splitting my data into testing and training data with a testing size of 0.2 and using data generators for data augmentation and supplying this enormous data to the model
#### actual building of the model
I am using transfer learning in which I am using vgg16 pre-trained model 
#### model architecture

the base model is the vgg16 model excluding the top 
and I included the following layers after it: 1.`global avg pooling` , 2. `dense(512,1024) layer`, 3.`dense(1024,1) layer` and 4.`a sigmoid`

#### Compiling the model

The optimizer I am using is `Adam optimizer`, loss function - `BCEloss` and `accuracy` for metrics

#### Training the model 
As I was using a Macbook It took 20 mins to run half a epoch 😭. even with this `the model had an accuracy of 89%`.
#### saving and loading model
I used model.save to save the model and load_model to load the model

### Deploying the attempt 2 model
I had a few problems while deploying because I didn't pass my images before the deployment to verify. So, I was getting an error when submitting so I had to run a few images locally and I found that the problem was the dimensions didn't match(I know noob mistake but this didn't strike my mind )So, I finally got the application to work. here are a few images for example (you can use the face-mask-dataset to test the model yourself!)the model is deployed at https://huggingface.co/spaces/saatvik1879/CVtask
![Alt text](/img1.jpg?raw=true "Optional Title")
![Alt text](/img2.jpg?raw=true "Optional Title")
![Alt text](/img3.jpg?raw=true "Optional Title")
![Alt text](/img4.jpg?raw=true "Optional Title")
