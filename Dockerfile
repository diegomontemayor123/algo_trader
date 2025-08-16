# Use Kaggle's Python image as base
FROM kaggle/python

# Set working directory
WORKDIR /kaggle/working

# Copy your project files (optional, you can still mount)
COPY . /kaggle/working

# Install required packages
RUN pip install ta
