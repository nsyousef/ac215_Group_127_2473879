# data-processor

The purpose of this microservice is to convert the raw datasets into a finalized format for processing.

## Building and Running Docker Image

### Build Image
```bash
docker build -t data-processor -f ac215_Group_127_2473879/src/data-processor/Dockerfile .
```

### Run Container
```bash
docker run --rm -ti data-processor
```

### Notes
- Run commands from root directory (one level above repo)
- Ensure `secrets/` folder exists in root
