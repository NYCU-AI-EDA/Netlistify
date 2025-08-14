#!/bin/bash
# find * -size +1M | cat >> .gitignore
# find * -size +99M | while read -r file; do
#     git lfs track "$file"
# done
# cp .gitignore .gitignore.bak
# find * -size +49M | cat >> .gitignore
git add -A
git commit -m "Auto commit $(date +%H/%M/%m/%d/%Y)"
# rm .gitignore
# mv .gitignore.bak .gitignore
git push -f
