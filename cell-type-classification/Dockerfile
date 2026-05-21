# ---------- Base image ----------
FROM python:3.8-bullseye
ENV DEBIAN_FRONTEND=noninteractive

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install pip for Python 3.8 ----------
RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

# ---------- Set python and pip alternatives ----------
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip 1

# ---------- Working directory ----------
WORKDIR /opt/pystruct

# ---------- Clone pystruct (includes AD3) ----------
RUN git clone https://github.com/pystruct/pystruct.git . \
    && sed -i '/use_2to3/d' setup.py

# ---------- Python dependencies ----------
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        "numpy==1.21.6" \
        scipy \
        "scikit-learn<2.0" \
        lightgbm \
        networkx \
        cvxopt \
        lxml \
        pandas \
        odfpy \
        openpyxl \
        xlrd \
        ezodf \
        python-dateutil \
        tqdm \
    && pip install cython

# ---------- Build & install pystruct (with AD3) ----------
RUN pip install --no-build-isolation --force-reinstall . \
    && pip install --force-reinstall "numpy==1.21.6"
# Re-force numpy install because pystruct uses deprecated functions

# ---------- Runtime patch for missing pystruct.models.utils ----------
RUN echo "import sys, types\nimport numpy as np\nmod = types.ModuleType('pystruct.models.utils')\n\ndef loss_augment_unaries(unary_potentials, y, class_weight):\n    n_states = unary_potentials.shape[1]\n    for i in range(unary_potentials.shape[0]):\n        yi = y[i]\n        for s in range(n_states):\n            if s == yi:\n                continue\n            unary_potentials[i, s] += class_weight[yi]\n\ndef crammer_singer_joint_feature(X, Y, out):\n    for i in range(X.shape[0]):\n        y = Y[i]\n        for j in range(X.shape[1]):\n            out[y, j] += X[i, j]\n\nmod.loss_augment_unaries = loss_augment_unaries\nmod.crammer_singer_joint_feature = crammer_singer_joint_feature\nsys.modules['pystruct.models.utils'] = mod" > /opt/pystruct/pystruct_utils_patch.py

# Automatically apply patch in every Python session
ENV PYTHONSTARTUP=/opt/pystruct/pystruct_utils_patch.py

# ---------- Project sources ----------
WORKDIR /opt/cell_type_classification
COPY . .

# ---------- Default command ----------
CMD ["python"]

