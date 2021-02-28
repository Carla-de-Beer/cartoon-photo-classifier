FROM tensorflow/serving:latest

COPY . /app

WORKDIR /app

EXPOSE 8500

ENTRYPOINT ["tensorflow_model_server", "--model_base_path=/app/models/cartoons-photos", "--model_name=cartoons-photos"]

CMD ["--rest_api_port=8501","--port=8500"]