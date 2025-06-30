TORCH_VERSION=${1:-"2.6.0"}
CUDA_VERSION=${2:-"124"}

# set -e # stop on error

echo "Installing flips with torch $TORCH_VERSION and cuda $CUDA_VERSION"

# wait for 3 seconds in case the user wishes to cancel the installation
sleep 3

THISDIR=$(dirname "$(readlink -f "$0")")

pushd "${THISDIR}/../.."

# install torch
pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cu$CUDA_VERSION
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html

# add the torch version to the requirements file to make sure it is not overwritten
cp flips/install_utils/requirements.txt flips/install_utils/tmp_requirements.txt
echo -e "\ntorch==$TORCH_VERSION" >> flips/install_utils/tmp_requirements.txt

# install pypi dependencies:
pip install -r flips/install_utils/tmp_requirements.txt

rm flips/install_utils/tmp_requirements.txt

# install gafl from source:
# Note: this is a temporary solution until gafl is available on pypi
git clone https://github.com/hits-mli/gafl.git
pushd gafl
bash install_gatr.sh # Apply patches to gatr (needed for gafl)
pip install -e . # Install GAFL
popd

git clone https://github.com/graeter-group/backflip.git
pushd backflip
pip install -e . # Install BackFlip
popd

# Finally, install flips:
cd flips
pip install -e . # Install flips

popd
