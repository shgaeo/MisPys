Para que los módulos en esta carpeta se puedan abrir entonces debes agregar la línea siguiente en el archivo ~/.bashrc

# para mis módulos
export PYTHONPATH="${PYTHONPATH}:/home/santiago/Documentos/MisPys"



Instrucciones para commit+push (https://dont-be-afraid-to-commit.readthedocs.io/en/latest/git/commandlinegit.html)
>>>git status                           ### para ver estatus
>>>git add file.name                    ### para pre-salvar archivo (-A para todos)
>>>git commit -m "short description"    ### commit para salvar cambios
>>>git push origin master               ### upload 
