# EDA

This file provides documentation on the EDA performed here.

To add dependencies, add them to `requirements.in`. Then, run `uv pip compile requirements.in > requirements.txt` to install them.

To set up the existing environment, run the following in the `eda` folder:

```
python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

To run Jupyter notebooks using the virtual enviornment in the `eda` directory, you may need to run:

```
python -m ipykernel install --user --name eda-venv --display-name "Python (eda venv)"
```

## How charging works for Google Cloud

* We are charged for storing data in the buckets. 
* We are also charged for transferring data out of the buckets (e.g. for EDA).
* If we transfer the data to a Google Cloud virtual machine in the same region as our bucket, I think the data transfer is free. However, we have to pay for the time we have the VM running.
* **It is generally much cheaper to do EDA in a virtual machine and pay for the time it is running than it is to transfer the data to our local machines and do EDA there.**
