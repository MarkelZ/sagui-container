# Set python 3.7
FROM python:3.7

# Set up working directory
WORKDIR /sagui/
COPY . /sagui/
COPY ./sagui/ /sagui/

# Create virtual environment
RUN python3.7 -m venv venv
ENV PATH="/sagui/venv/bin:$PATH"

# Copy .mujoco to docker user's home directory
COPY .mujoco/ /root/.mujoco/

# Install required libraries
RUN apt-get update && apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf apt-utils

# Export library variable for mujoco_py
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# Install project
RUN pip install -e .

# Set entrypoint
COPY entrypoint.sh /sagui/entrypoint.sh
RUN chmod +x /sagui/entrypoint.sh
RUN chmod -R 755 /sagui
ENV PYTHON_SCRIPT train-guide.py
ENTRYPOINT ["/sagui/entrypoint.sh"]

