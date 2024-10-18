# pytorch_xpu_play

Illustrates use of PyTorch 2.5 with native XPU support, as documented at https://pytorch.org/docs/main/notes/get_start_xpu.html, https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html, and https://dgpu-docs.intel.com/driver/client/overview.html#installing-client-gpus-on-ubuntu-desktop-24-04-lts

It has been tested on a 13th Gen Intel(R) Core(TM) i9-13900K processor with Intel(R) ARC A770 Discrete GPU

## Docker 

These samples utilize containers to fully encapsulate the example with minimial host dependencies.  Here are the instructions how to install docker:

```
$ sudo apt-get update
$ sudo apt-get install ca-certificates curl
$ sudo install -m 0755 -d /etc/apt/keyrings
$ sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
$ sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install docker
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable docker access as user
sudo groupadd docker
sudo usermod -aG docker $USER
```
Note: This configuration above grants full root access of the container to your machine. Only follow this if you understand the implications for doing so, and don't follow this procedure on a production machine.

## Usage

To build the container, type:
```
$ ./build
```

To execute the container and run the PyTorch 2.5 sample inference and training examples:
```
$ ./run
```
