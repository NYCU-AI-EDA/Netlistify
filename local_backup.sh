# rsync -vzhr --delete --progress ../ViT-Schematic server7:/home/deathate/Projects
rsync -vzhr --progress ../ViT-Schematic server5:/home/112/deathate/Projects --exclude-from='exclude-list.txt'
# rsync -vzhr  --progress server7:/home/deathate/Projects/ViT-Schematic/tmp . --dry-run
# rsync -vzhr --progress ../ViT-Schematic hank@server5:/home/111/hank/img2hspice/demo/ --exclude-from='exclude-list.txt' --dry-run