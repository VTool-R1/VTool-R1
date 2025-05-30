gdown https://drive.google.com/drive/folders/1Ic2BmpbGQ1pcZ6KabjmP9YxefKDq3TrN --folder
mv ReFocus_data data
cd data
for zip in *.zip; do unzip -o "$zip"; done