# Documentación Modelos

En este documento se tiene el propósito de informar los detalles relacionados con los **modelos de Inteligencia Artificial** implementados para el proyecto **CatVision**, así como la relación que cada uno tiene con los demás modelos, sus **propósitos** específicos y el **orden** en el que fueron utilizados. Estos modelos en conjunto buscan lograr **identificar** dentro de un refrigerador con especificaciones controladas los productos pertenecientes a cada una de las distribuidoras que han contratado el servicio con el fin de proveer esta información a los servidores de **CatVision**.

**Pipeline**

El esquema de trabajo que se sigue para ensamblar los resultados de los modelos entre sí es el siguiente:

[https://lh5.googleusercontent.com/2Skbgd1MQTbjf0tMYdfoh6CMVUdGmn8yzNHJ0iYTvf5GST52V89c0lYwgTSklMwi3jeyMuEO3oSEy6bOllBoZwf_zQ7bV3UMq_3CukJ0PyJ1vdCqYTobwsuixObnX_1y1L4VW27NHidVSPknXBNRUEhVncHs7qhUYC5P88aKLnafco1eeuBHsAUZbkUIEQ](https://lh5.googleusercontent.com/2Skbgd1MQTbjf0tMYdfoh6CMVUdGmn8yzNHJ0iYTvf5GST52V89c0lYwgTSklMwi3jeyMuEO3oSEy6bOllBoZwf_zQ7bV3UMq_3CukJ0PyJ1vdCqYTobwsuixObnX_1y1L4VW27NHidVSPknXBNRUEhVncHs7qhUYC5P88aKLnafco1eeuBHsAUZbkUIEQ)

La forma en la que cada uno de los cuatro modelos presentados en el diagrama trabaja es la siguiente:

1. **MS1 (*Modelo de Segmentación 1*): Este modelo se encarga de detectar el contorno del refrigerador y recortar la imagen de tal manera que únicamente esté presente el objeto con los productos en su interior.**
2. **MS2 (*Modelo de Segmentación 2*): Este modelo recibe la imagen del refrigerador y procede a detectar formar los contornos de los productos que podrían pertenecer a la distribuidora en cuestión, una vez que los encuentra, recorta únicamente el producto encontrado, si se detecta que hay un espacio potencialmente vacío, se envía a una carpeta y de ahí su etiqueta se envía al servidor.**
3. **MC (*Modelo de Clasificación*): Este es un ensamble de tres modelos que recibe las imágenes que contiene exclusivamente al producto y realiza una clasificación para lograr identificar cuál es el producto en cuestión.**

Una vez que el pipeline ha sido ejecutado completamente, la información de ambos modelos de clasificación es enviada al servidor para su posterior procesamiento.

**Especificaciones de los modelos:**

1. **MS1 (*Modelo de Segmentación 1*):**
    1. **Tecnología implementada: Detectron 2 (Transfer learning)**
2. **MS2 (*Modelo de Segmentación 2*):**
    1. **Tecnología implementada: Detectron 2 (Transfer learning)**
3. **MC (*Ensamble de* *Modelos de Clasificación*):**
    1. **Tecnología implementada: Tensor Flow.**
    2. **Estructuras implementadas: Un modelo VGG 16 (Modelo pre-entrenado para clasificación multi-categórica usando los pesos ImageNet), una red convolucional hecha a mano con 3 pares de capas convolutivas y de Max Pooling y dos capas densas, una con 128 neuronas y una con 21 (número de clases), y una última red convolutiva hecha en Teachable Machine.**
    3. **Técnicas de ensamble: Ensamble de 3 modelos convolutivos.**

**Tratamiento de los datos.**

Dado que se trata de un ambiente controlado, puesto que la mayor parte de las variables en juego son predominantemente constantes, el tratamiento de datos no requirió de una gran cantidad de **data augmentation**, ya que lo único que realmente tiene el potencial de ser variable es el **brillo**, el **contraste**, la **rotación**, y el **zoom** de las imágenes.

Así mismo, el tamaño del dataset se mantuvo en aproximadamente 20-30 imágenes por producto, sin embargo la porción de datos correspondientes a la **validación de resultados** se mantuvo en **20%**. Después, se añadió una capa de regularización de tamaño de imágenes, que en **keras** tiene el nombre de **rescaling**, esto se hizo con el fin de mantener los datos en una escala homogénea, permitiendo identificar más rápidamente el mínimo local durante el entrenamiento del modelo y que los resultados de los modelos de clasificación no se vean afectados. Así mismo, se añadió una capa de **data augmentation** que alteran el **brillo**, el **contraste**, la **rotación** y el **zoom** de las imágenes en un rango de **20%**, arrojando las siguientes gráficas de entrenamiento del **MC:**

[https://lh6.googleusercontent.com/p7ADDsEOGx--WKYCUMgnHvmLW8iH1SwdRthQYTXj25grRE_FuIz6ZVQc2jMHY90yhhN88hAMDoizk7LUBHuA0qUPuBGAYMRxHkKgXWDMbec4brGnjQZRgCsS6zfZDDnAb6haQMQQ9Gt6MsCPNWis5BFtihHH1J4s3287UkZkKQcQNpcMcWvWBSQhgwmEmw](https://lh6.googleusercontent.com/p7ADDsEOGx--WKYCUMgnHvmLW8iH1SwdRthQYTXj25grRE_FuIz6ZVQc2jMHY90yhhN88hAMDoizk7LUBHuA0qUPuBGAYMRxHkKgXWDMbec4brGnjQZRgCsS6zfZDDnAb6haQMQQ9Gt6MsCPNWis5BFtihHH1J4s3287UkZkKQcQNpcMcWvWBSQhgwmEmw)

Modelos de segmentación:

Para la primera parte de recortado que involucra, tanto al refrigerador, así como a las botellas, se utilizó un proceso de transfer learning, en donde se involucró un modelo llamado “Detectron 2” el cual básicamente es una librería de IA de facebook research, la cual cuenta con distintas configuraciones en archivos de tipo yaml, es importante resaltar que los modelos que se utilizan dentro de detectron 2 son modelos que cuentan con entrenamiento previo en distintas bases de datos de imágenes, pero a su vez, debido a que dichos modelos abarcan una gran cantidad de clases, realmente no están muy bien entrenados en clases en específico, lo que realmente dificulta utilizar dichas pre configuraciones en un problema en específico, afortunadamente dentro de la documentación se puede encontrar como entrenar tu propio modelo, refiriéndose solo a una clase de las que ya se tienen estipuladas e inclusive agregar nuevas, permitiendo de esta forma utilizar la información recolectada por el modelo, así como tus propias clases e imágenes de entrenamiento y validación.

Para las imágenes se utilizaron todas las recopiladas por el equipo a lo largo del semestre, además de una marcación de puntos claves dentro de las imágenes, rodeando los objetos de nuestro interés, realizando un etiquetado sobre los mismos, en un ambiente tanto controlado como uno natural, permitiendo que el modelo tenga diferentes perspectivas sobre los objetos, además de proporcionarle las clases a las cuales queremos asignarlos, dichas imágenes, previo al etiquetado y marcación de puntos pasaron por un pre procesado en el cual tambien se ajusto un tamaño estandar de imagen de 540 x 540, de forma que el modelo tuviera una métrica estándar y fuera mucho más preciso, además de agregar capas de data augmentation que permitieran distintos cambios de ambiente y/o posiciones para hacer que nuestro modelo fuera más adaptable a cualquier situación  posteriormente se dividieron en los sets de validación y entrenamiento.

Con esto se abarcó el entrenado de los dos modelos de segmentación ya que ambos tienen una arquitectura similar, siendo que lo único que realmente genera la diferencia, es el dataset de imágenes que se utilizó para el entrenamiento de cada uno, ya que en el del refrigerador, se utilizaron imágenes de refrigeradores, mientras que en el de botellas y latas se utilizaron imágenes de las mismas.

Aquí se muestra una imagen de nuestra loss function a lo largo del entrenamiento del modelo:

[https://lh3.googleusercontent.com/T7y6ASYbDmxiqcq4BdXFqEAt_g-nBqvt0MpNxMNdmZpTcR3bbv-9Zh6jVEs9sYzNISJJQvOHttyDFkj5A4sijfbfxrgHf4RfGj58aY4HG83-KOE4FKqka78VLEafHsiDcfVp2mkVWRDkxRQLOAkcV7Ocs24meLMXaWhOi9qRXGso4SMEou4sxBtTfhKvKg](https://lh3.googleusercontent.com/T7y6ASYbDmxiqcq4BdXFqEAt_g-nBqvt0MpNxMNdmZpTcR3bbv-9Zh6jVEs9sYzNISJJQvOHttyDFkj5A4sijfbfxrgHf4RfGj58aY4HG83-KOE4FKqka78VLEafHsiDcfVp2mkVWRDkxRQLOAkcV7Ocs24meLMXaWhOi9qRXGso4SMEou4sxBtTfhKvKg)

Teniendo en cuenta que también nuestro modelo para nuestro ambiente en específico, con el entrenamiento otorgado y la validación que se realizó, en nuestra situación en particular tiene un accuracy promedio de alrededor de un 97% en la mayoría de situaciones en donde se ha probado, evidentemente dependiendo de la situación presentada este porcentaje puede variar ya sea mayor o menor según sea el caso.

[https://lh4.googleusercontent.com/nuW6zirrXvVYBo6Um-yApkT8MesjaRwEKgJSlMWdSUFibbQFcndtGQNZyu6tUWP1gAcBO1UHcn9C0pZHH40__9rwJPaXWZ4vOZckhZ6jH7sMF6jRk4UC-2M2E_boQ2wFnJQmKhNO_4WTpdI-9p-xVxCFR7-8uWPk-YBwDJVvXIezodsIpNdUmXOtjm-U6g](https://lh4.googleusercontent.com/nuW6zirrXvVYBo6Um-yApkT8MesjaRwEKgJSlMWdSUFibbQFcndtGQNZyu6tUWP1gAcBO1UHcn9C0pZHH40__9rwJPaXWZ4vOZckhZ6jH7sMF6jRk4UC-2M2E_boQ2wFnJQmKhNO_4WTpdI-9p-xVxCFR7-8uWPk-YBwDJVvXIezodsIpNdUmXOtjm-U6g)

[https://lh3.googleusercontent.com/N-OP4XtmGR-l0o_pMOsw8WgNyOSF-w9_O2sPTETNWx0FfkLBAsAsAQYixyNKMS6RFqd18gB7yK02RLjUyzFABxEjSJn2EF05qCrclrwHaO6IkBnAYInuqFvSs1eOCUkB0n4BXj-294uDO8SrHu18vtluxvwQVZIqe_jOiPTRkvfI_ntyGpVfFMTE7aih4g](https://lh3.googleusercontent.com/N-OP4XtmGR-l0o_pMOsw8WgNyOSF-w9_O2sPTETNWx0FfkLBAsAsAQYixyNKMS6RFqd18gB7yK02RLjUyzFABxEjSJn2EF05qCrclrwHaO6IkBnAYInuqFvSs1eOCUkB0n4BXj-294uDO8SrHu18vtluxvwQVZIqe_jOiPTRkvfI_ntyGpVfFMTE7aih4g)

- [Modelos de clasificación.](https://colab.research.google.com/drive/1WafrMYJs4XIZPMHBAT2FyDLPJjKdMwCB?usp=sharing)
- [Modelo de segmentación 1.](https://colab.research.google.com/drive/11QLOjc_IVeJ99UY16XxIXiR8HsNhn_Eu?usp=sharing)
- [Modelo de segmentación 2.](https://colab.research.google.com/drive/1Vtaufi0IRvvLzdb2BI2QHCO2fiUEDdVS?usp=sharing)