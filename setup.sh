sudo apt update
sudo apt upgrade
sudo apt install build-essential libffi-dev libssl-dev zlib1g-dev liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev tk-dev libmpv-dev libmpv2
sudo ln -s /usr/lib/x86_64-linux-gnu/libmpv.so /usr/lib/libmpv.so.1
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo '' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
source ~/.bashrc
pyenv install 3.11
pyenv global 3.11
cd ~/HandsOnWorkshop/fletCVPR/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
