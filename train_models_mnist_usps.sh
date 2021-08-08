wget https://github.com/rll/deepul/raw/master/homeworks/hw1/data/hw1_data.zip
unzip hw1_data.zip
mkdir ./data
mkdir ./plots
mv shapes.pkl ./data/shapes.pkl
mv shapes_colored.pkl ./data/shapes_colored.pkl
mv mnist.pkl ./data/mnist.pkl
mv mnist_colored.pkl ./data/mnist_colored.pkl
rm hw1_data.zip
rm geoffrey-hinton.jpg
rm smiley.jpg

python train_models.py