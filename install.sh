cp -R ./libs $1
cp -R ./data $1
cp -R ./notebooks $1
cp -R ./paper $1
cp -R ./workflow $1
cp ./.gitignore $1
cp ./Snakefile $1
cp ./create_environment.sh $1
touch $1/README.md
