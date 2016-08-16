sudo apt-get update
sudo apt-get -y install python-pip python-dev python-virtualenv libblas-dev liblapack-dev libatlas-base-dev gfortran
virtualenv --system-site-packages ~/virtual-tf
pip install dask[dataframe]

pip install scipy

pip install matplotlib

pip install pymatbridge

pip install scikit-learn

sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl

git clone --recurse-submodules https://github.com/tensorflow/tensorflow

git clone --recurse-submodules https://github.com/tensorflow/tensorflow

echo "Make a quick link to udacity example files" 

ln -s /home/ubuntu/workspace/tensorflow/tensorflow/examples/udacity /home/ubuntu/workspace/udacity-Link


echo "Make a quick link to skflow example files" 

ln -s /home/ubuntu/workspace/tensorflow/tensorflow/examples/skflow /home/ubuntu/workspace/skflow-examples/skflow-link

echo "Make a link to the actual installed pip tensorflow not just the github clone"

ln -s /home/ubuntu//virtual-tf/lib/python2.7/site-packages/tensorflow /home/ubuntu/workspace/pip-tensorflow-link

pip install --upgrade ipython
pip install --upgrade jupyter



jupyter notebook --ip $IP --port $PORT --no-browser


wget www.googledrive.com/host/0B_u1P2oANsMaX2o1azZKS0FFYkE
mv 0B_u1P2oANsMaX2o1azZKS0FFYkE mypickle.zip
unzip mypickle.zip
