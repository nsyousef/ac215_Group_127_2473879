# ac215_Group_127_2473879

## File Structure

`eda`: stores exploratory data analysis for our project
`src`: stores the source code for our project
    
    TODO: describe each folder in src


## Building and Running Docker Images

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
