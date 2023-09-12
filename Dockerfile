################################################################
# safety-gym
################################################################

# Set python 3.7
FROM python:3.7

# Create a virtual environment
RUN python3.7 -m venv venv
ENV PATH="/safety_gym/venv/bin:$PATH"

# Copy mujoco to docker user's home directory
COPY ./mujoco/ /root/.mujoco/

# Install required libraries
RUN apt-get update && apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf apt-utils libopenmpi-dev 

# Export library variable for mujoco_py
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# Install Python package dependencies
COPY requirements1.txt /
RUN pip install -r requirements1.txt

COPY requirements2.txt /
RUN pip install -r requirements2.txt

# Set up working directory
WORKDIR /safety_gym/
COPY ./safety-gym/ /safety_gym/
COPY ./safety-gym/safety_gym/ /safety_gym/

# Install project
RUN pip install -e .

# Set up working directory
WORKDIR /sagui/
COPY ./SaGui/ /sagui/
COPY ./SaGui/sagui/ /sagui/

# Install project
RUN pip install -e .

# Use a shell as the entry point
ENTRYPOINT ["/bin/bash"]
