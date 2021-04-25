<h1> Task 3: Modifying Model </h1>

<strong>*Full documentation is done in Task3Modification.ipynb notebook</strong>

Now, we try to leverage on the training speed that MobileNet-V2 provides as well as its respectable accuracy results.

However, in our specific detection objective of pedestrains from the PennFudan dataset, MobileNet does not seem to work as well.

<h2> Examples </h2>

<h4> Example 1 </h4>

In this example, both models seems to produce similar results

           Original               Resnet50               Mobilenet

<p align="center">
  <img src="https://user-images.githubusercontent.com/68470272/92985448-21348680-f4e5-11ea-9f08-c99e3a36b359.png" width="300">
  <img src="https://user-images.githubusercontent.com/68470272/93066948-2ba17c80-f6ad-11ea-82f8-2f4f3af60104.png" width="300">
  <img src="https://user-images.githubusercontent.com/68470272/93067651-124d0000-f6ae-11ea-8dee-1cc7dccbfa6c.png" width="300">
</p>

<h4> Example 2 </h4>

However, in this image where there is much more objects to be detected, it is quite clear that the model trained using resnet50 is much more precise in its detection. It is able to detect more pedestrains with a higher accuracy compare to mobilenet.


              Original            Resnet50            Mobilenet
<p align="center">
  <img src="https://user-images.githubusercontent.com/68470272/92985505-ba639d00-f4e5-11ea-8333-b3e1d4294f20.png" height="240">
  <img src="https://user-images.githubusercontent.com/68470272/93066940-29d7b900-f6ad-11ea-9061-59f91ab1dfa8.png" height="240">
  <img src="https://user-images.githubusercontent.com/68470272/93067781-3c062700-f6ae-11ea-9f16-249d3d827ef3.png" height="240">
</p>
