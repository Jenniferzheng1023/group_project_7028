# group_project_7028

## Step 1

```shell
cd ./ui
```

## Step 2
In stall requirement package

```shell
pip install requirements.txt
```

You also need to install `tkinter` (cannot use pip to install)
For mac
```shell
$ brew install python-tk
```
After use the brew to install but not work for other package like pillow. You can try following to create the venv environment

```shell
python3 -m venv myenv # create the venv called myenv
```

```shell
source myenv/bin/activate #activate the venv
```
``` shell
python3 -m pip install requirements.txt #try to reinstall the require package
```

For linux
```shell
$ sudo apt-get install python3-tk
```

## Step 3

Run the code
```shell
python3 tk_test.py
```

![Current UI Demo](/demo.png "Current UI Demo")

