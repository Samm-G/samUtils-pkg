# Samutils python package

## How to use this package
```python
from samutils.perceptron import Perceptron
model = Perceptron(eta=0.3, epochs=10)
```

## Setup : 
### Create your own env:
```
conda create --prefix "./conda_env" python==3.8 -y

conda activate "./conda_env"

pip cache purge

pip install -r requirements.txt
```

### Recreate your own Git:
```
rm -rf .git

git init

git add .

git commit -m "First Commit"

git branch -M main

git remote add origin <Your New Repository>

git push -u origin main

```

## Upload to PyPI:
```
Get LICENSE from Github > Add File > Create New File > Type "LICENSE" and click on "Choose A Template"

Local:
    git pull origin main

Go to PyPI.com > Account Settings > Add API Token
Give Token name
    Scope: (Entire Account)
Go to Github Repo > Settings > Secrets
    Add a Secret:
        Name: PYPI_API_TOKEN
        Value: (Copied from Pypi)

Trigger the Actions Pipeline to Push to PyPI:
    git push origin main
```

## References - 

* [Official python docs for PYPI](https://packaging.python.org/tutorials/packaging-projects/)

* [Github action file for PYPI](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python#publishing-to-package-registries)