# Project Setup

## Create a Python Virtual Environment

1. **Create a virtual environment:**
python3 -m venv venv
2. **Activate the virtual environment:**
   - On Linux/macOS:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```
3. **Upgrade pip:**
```
    pip install --upgrade pip
```
4. **Install dependencies:**
```
    pip install -r requirements.txt
```

5. **Test setup the application:**
```
python inference.py
```
6. **Deactivate the virtual environment (when done):**
```
deactivate
```


## Sample Usage

```
    img = cv2.imread('data/sample.jpg')
    result = detect_potatos(img)
    print('Potato Coordinates for data/sample.jpg are:', result)
```

This would output something like:
```
[(4, 2), (6, 4)]
indicating the coordinates of detected potatoes in the image.
```
