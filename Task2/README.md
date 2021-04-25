<h1>Mask RCNN API</h1>

<h2>User Guide</h2>

*<strong>Testings are done in Python version 3.7</strong>

1. Create models folder in `Task2` directory with `mkdir models`. Download trained model using the link below and save it into this new directory.
Our script will look into the folder `models` when predicting on the image posted by user.
> https://drive.google.com/file/d/1zRpg1zLRtHRmjVWAohc3iXId4M5GXeAr/view?usp=sharing
2. Run the API built using Flask.

    `python3 maskAPI.py`

3. The server would be running and you can POST an image to the server. For example, using curl
    
    `curl -F 'image=@FILE_PATH_OF_IMAGE' 'http://127.0.0.1:5000/upload'`

4. Or using Python

    ```
    import requests
    url = 'http://127.0.0.1:5000/upload'
    with open("FILE_PATH_OF_IMAGE","rb") as img:
        files = {"image": (url,img)}
        with requests.Session() as s:
		        r = s.post(url,files=files)
		        print(r.json())
    ```
> If you posted an image file, you will get back an url. Simply append this to the server. Default server is http://127.0.0.1:5000/

<h2>Reflections </h2>

Looking at this task initially, it appeared to be straightfoward. Build an API that accepts POST request of image, feed it into a model that is already
trained, and then return this image to the API user. However, questions like - how do I even send this image back to the user, perhaps needing an
interface to display this image? Futhermore, the idea of creating an API was unclear to me, despite being an avid API user, but have never experienced being
on the other side.

Ultimately, I decided to save the new image onto the server first, and then allow the user to access it through the url. Currently,
the images are not deleted unless a new image with the same filename as existing ones are uploaded (in which it would be overwritten). In production,
depending on whether this data needs to be stored, we can handle this differently such as simply overwriting all new images and maintaining only one in the server.

The use of API wrapper flask_restful has simplified my code and also is able to handle the correct image file received by itself without explitcitly doing so. One
possible improvement could be to dynamically receive the file and then send it back to the user without needing to write it to server at all.

When loading the saved model, it is important to not make any changes to the artitecture of our neural network that we trained our models on.

One thing that took me some time to figure out was to calculating percent_masked of the image. If we just try to find the number of zeroes in the sequence of
tensors of all masks, we would be overcounting as there could be pixels overlapping between different masks. Eventually, I found out about PyTorch built in
Tensor().Add which can add values of tensors index wise. So this way, we can count zeroes only after summing all masks together.
