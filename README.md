# keras-vision-app

Deployment ready starter pack for creating fast responsive Web App for Keras or Tensorflow2 Vision models using Starlette.io framework with Uvicorn ASGI server.

Everything packaged in docker with requirement.txt, so you can push it to any docker hosted cloud service. Enjoy :)

Also, You can test your changes locally by installing Docker and using the following command:

docker build -t keras-vision-app . && docker run --rm -it -p 8080:8080 keras-vision-app

Few dockers hosted services where this starter pack should work =>

* https://render.com
* https://zeit.co/now
* https://azure.microsoft.com/en-us/services/app-service/containers/
* https://getcarina.com/
* https://sloppy.io/en/
* https://giantswarm.io
* https://aws.amazon.com/ecs/
* https://cloud.google.com/cloud-build/docs/
* https://www.digitalocean.com/products/one-click-apps/docker/
