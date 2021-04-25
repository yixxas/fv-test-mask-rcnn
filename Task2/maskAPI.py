from flask import Flask, url_for
from flask_restful import Api, Resource, reqparse
import werkzeug
import base64
from PIL import Image
from io import BytesIO
import cv2
from customModel import getResults

app = Flask(__name__)
api = Api(app)


class GenerateImage(Resource):
	def __init__(self):
		parser = reqparse.RequestParser()
		parser.add_argument("image", type=werkzeug.datastructures.FileStorage, location='files')
		self.req_parser = parser
	
	def post(self):
		image_file = self.req_parser.parse_args(strict=True).get("image", None)

		imageFilename = (image_file.filename)

		if image_file:
			print("Processing Image .................")

			image = Image.open(image_file).convert('RGB')
			size,img_cv = getResults(image)
			#saving image onto server and overwriting each time
			cv2.imwrite("static/"+imageFilename,img_cv)
			return {"output":url_for('static', filename = imageFilename),"percent_masked": float("{:.2f}".format(size*100))}
		else:
			return "Not an image"
api.add_resource(GenerateImage, "/upload")


if __name__ == "__main__":
	app.run(debug=True)


