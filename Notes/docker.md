$ sudo apt install docker-ce docker-ce-cli containerd.io
$ sudo docker run hello-world
$ docker pull postgres
$ docker run --name main_postgres -e POSTGRES_PASSWORD=mysecretpassword -d postgres
docker compose up -d в дирректории где лежит compose.yml файл d
docker compose watch #Чтобы изменения из файлов сразу отражалить в запущенном контейнером сервере

docker build -t myimage . Собирает имедж по файлу

--restart отвечает за рестарты при ощибках

Варианта мультипликации 2: 1. Кубер и контейнер на каждое ядро 2. Gunicorn с несколькими воркерами

1. pip install "uvicorn[standard]" gunicorn
   gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:80
