# Set python 3.7
FROM python:3.7

# Create a virtual environment
RUN python3.7 -m venv venv

# Create and switch to a non-root user
RUN useradd -ms /bin/bash dockeruser
USER dockeruser

# Copy repo to the container
WORKDIR /sagui-container/
COPY . /sagui-container/

# Copy mujoco to docker user's home directory
RUN mv /sagui-container/mujoco/ /root/.mujoco/

# Install required libraries
RUN apt-get update && apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf apt-utils libopenmpi-dev 

# Export library variable for mujoco_py
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# Install Python package dependencies
COPY requirements1.txt /
RUN pip install -r requirements1.txt

COPY requirements2.txt /
RUN pip install -r requirements2.txt

# Install safety-gym
WORKDIR /sagui-container/safety-gym/
RUN pip install -e .

# Install SaGui
WORKDIR /sagui-container/SaGui/
RUN pip install -e .

# Switch to main working directory
WORKDIR /sagui-container/SaGui/sagui

# Use a shell as the entry point
ENTRYPOINT ["sh", "-c", "git fetch && git pull && /bin/bash"]
