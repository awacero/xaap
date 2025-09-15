# XAAP — Documento de análisis técnico

## 1. Propósito y alcance
Este documento complementa la especificación de requerimientos describiendo cómo la solución XAAP puede satisfacerlos desde una
perspectiva técnica. Se detallan las interacciones entre actores, módulos principales, dependencias externas y estructuras de
datos producidas por la tubería de adquisición, detección y clasificación sísmica. La intención es proveer una base común para
el diseño detallado, la estimación de esfuerzo y la planificación de iteraciones futuras.

## 2. Contexto de la solución
XAAP orquesta una secuencia de pasos para automatizar el análisis de eventos volcánicos:

1. **Configuración**: la interfaz (GUI o CLI) construye un objeto `xaapConfig` consolidando endpoints mSEED, catálogos de
   estaciones, filtros, parámetros STA/LTA, metadatos de modelos SeisBench y rutas de salida.
2. **Adquisición**: `request_data.request_stream` consulta servicios mSEED (vía `get_mseed_data`) para descargar y unificar
   trazas en un `Stream` por estación.
3. **Preprocesamiento**: `pre_process.pre_process_stream` aplica detrending, merge y filtros de frecuencia según la
   configuración.
4. **Detección**: la etapa clásica usa `detect_trigger.get_triggers` (STA/LTA + coincidencia); la variante profunda emplea
   `process_deep_learning.get_detections` y `DetectionXaap` para transformar inferencias SeisBench.
5. **Clasificación**: `classify_detection.classify_detection_SVM` extrae atributos con `aaa_features` y ejecuta un modelo
   scikit-learn para etiquetar detecciones.
6. **Revisión**: las herramientas de revisión manual consumen CSV de detecciones y clasificaciones para validación humana.

## 3. Actores y casos de uso
### 3.1 Diagrama de casos de uso
```plantuml
@startuml
left to right direction
actor "Operador GUI" as GUI
actor "Operador CLI" as CLI
actor "Analista de revisión" as Reviewer
actor "Administrador de modelos" as ModelAdmin

usecase "Configurar parámetros\ny conexiones" as UC_Config
usecase "Solicitar y preparar\nondas" as UC_Data
usecase "Detectar eventos\nSTA/LTA" as UC_STA
usecase "Detectar eventos\nDeep Learning" as UC_DL
usecase "Clasificar detecciones" as UC_Class
usecase "Guardar y compartir\nresultados" as UC_Export
usecase "Revisar manualmente\ndetecciones" as UC_Review
usecase "Gestionar modelos\ny características" as UC_ModelMgmt

GUI --> UC_Config
GUI --> UC_Data
GUI --> UC_STA
GUI --> UC_DL
GUI --> UC_Class
GUI --> UC_Export
GUI --> UC_Review

CLI --> UC_Config
CLI --> UC_Data
CLI --> UC_STA
CLI --> UC_DL
CLI --> UC_Class
CLI --> UC_Export

Reviewer --> UC_Review
ModelAdmin --> UC_ModelMgmt
UC_ModelMgmt --> UC_DL
UC_ModelMgmt --> UC_Class
UC_Export <-- UC_Review
@enduml
```

### 3.2 Descripción resumida de casos de uso
- **UC_Config**: seleccionar volcán, estaciones, fechas, filtros y modelos disponibles. La configuración puede persistirse como
  JSON para ejecuciones futuras.
- **UC_Data**: solicitar trazas al servicio mSEED seleccionado, consolidarlas por estación y preparar el flujo para etapas
  posteriores.
- **UC_STA**: ejecutar coincidencia STA/LTA con umbrales configurables y generar un reporte de detecciones.
- **UC_DL**: cargar el modelo SeisBench apropiado, inferir picks/detecciones y convertirlos en ventanas XAAP.
- **UC_Class**: generar vectores de características y aplicar el clasificador scikit-learn activo.
- **UC_Export**: escribir archivos CSV con resultados de detección/clasificación y bitácoras asociadas.
- **UC_Review**: abrir CSVs resultantes, recuperar ondas asociadas y permitir validación/corrección humana.
- **UC_ModelMgmt**: administrar versiones de modelos SeisBench y clasificadores, además de catálogos de características.

## 4. Vista dinámica del proceso principal
### 4.1 Diagrama de secuencia (ejecución típica)
```plantuml
@startuml
actor Operador
participant "GUI/CLI" as UI
participant "xaapConfig" as Config
participant "request_data" as Request
participant "pre_process" as Preprocess
participant "detect_trigger" as StaLta
participant "process_deep_learning" as Deep
participant "classify_detection" as Classify
participant "Almacenamiento CSV" as Storage

Operador -> UI : Define parámetros / inicia ejecución
UI -> Config : __init__(xaap_parameter)
Config --> UI : instancia configurada
UI -> Request : request_stream(config)
Request --> UI : volcan_stream
UI -> Preprocess : pre_process_stream(config, volcan_stream)
Preprocess --> UI : stream_preprocesado

alt Método STA/LTA
  UI -> StaLta : get_triggers(config, stream_preprocesado)
  StaLta --> UI : triggers
else Método deep learning
  UI -> Deep : get_detections(config, stream_preprocesado)
  Deep --> UI : detections
end

opt Clasificación solicitada
  UI -> Classify : classify_detection_SVM(config, stream_preprocesado, detections)
  Classify --> Storage : CSV clasificación
end
UI --> Storage : CSV detecciones
@enduml
```

### 4.2 Consideraciones
- El flujo incluye validaciones de disponibilidad (streams, modelos, rutas) antes de ejecutar cada etapa.
- Los mensajes de log acompañan cada transición para seguimiento en tiempo real.
- La revisión manual ocurre posteriormente consumiendo los archivos generados.

## 5. Vista lógica (clases y dependencias)
### 5.1 Diagrama de clases simplificado
```plantuml
@startuml
hide empty members
skinparam classAttributeIconSize 0

class "xaap.configuration.xaapConfig" as XaapConfig {
  +mseed_client_id
  +volcanoes_stations
  +datetime_start
  +filter_freq_a
  +sta_lta_sta
  +deep_learning_model_name
  +output_detection_folder
  +classification_model_file
  +__init__(xaap_parameter)
}

class "xaap.process.request_data" as RequestData <<module>> {
  +request_stream(xaap_config)
}

class "xaap.process.pre_process" as PreProcess <<module>> {
  +pre_process_stream(xaap_config, stream)
}

class "xaap.process.detect_trigger" as DetectTrigger <<module>> {
  +get_triggers(xaap_config, stream)
  +coincidence_trigger_deep_learning(...)
}

class "xaap.process.process_deep_learning" as DeepProcess <<module>> {
  +get_detections(xaap_config, stream, model)
  +coincidence_detection_deep_learning(...)
}

class "xaap.process.detection_xaap.DetectionXaap" as DetectionXaap {
  +trace_id
  +start_time
  +end_time
  +pick_detection
  +__lt__(other)
}

class "xaap.process.classify_detection" as ClassifyDetection <<module>> {
  +classify_detection_SVM(xaap_config, stream, detections)
}

class "xaap_gui" as Gui <<script>>
class "xaap_cli" as Cli <<script>>
class "xaap_manual_check" as ManualCheck <<script>>

Gui --> XaapConfig
Cli --> XaapConfig
Gui --> RequestData
Cli --> RequestData
Gui --> PreProcess
Cli --> PreProcess
Gui --> DetectTrigger
Cli --> DetectTrigger
Gui --> DeepProcess
Cli --> DeepProcess
Gui --> ClassifyDetection
Cli --> ClassifyDetection

DeepProcess --> DetectionXaap
ClassifyDetection ..> DetectTrigger
ManualCheck --> XaapConfig : reutiliza catálogos
ManualCheck --> RequestData : recupera ondas
@enduml
```

### 5.2 Notas sobre la vista lógica
- `xaapConfig` centraliza rutas, parámetros físicos y banderas de ejecución compartidas por todos los módulos.
- Los módulos de proceso operan con funciones puras que aceptan `Stream` y devuelven listas/objetos; esto facilita pruebas
  unitarias aisladas.
- `DetectionXaap` actúa como contenedor común para detecciones generadas por modelos profundos y permite ordenarlas.

## 6. Vista de componentes y dependencias externas
```plantuml
@startuml
skinparam componentStyle rectangle
component "GUI PyQt\n(xaap_gui.py)" as GUI
component "CLI\n(xaap_cli.py)" as CLI
component "Herramienta revisión\n(xaap_manual_check.py)" as Review
component "Pipeline de proceso" as Pipeline
component "Servicios mSEED\n(get_mseed_data)" as Mseed
component "SeisBench / PyTorch" as SeisBench
component "Biblioteca de características\n(aaa_features)" as Features
component "Modelos scikit-learn" as Sklearn
component "Sistema de archivos" as Files

GUI --> Pipeline
CLI --> Pipeline
Review --> Pipeline : lectura selectiva

Pipeline --> Mseed : solicitud de streams
Pipeline --> SeisBench : inferencia deep learning
Pipeline --> Features : extracción de atributos
Pipeline --> Sklearn : clasificación
Pipeline --> Files : CSV detecciones / logs
Review --> Files : lectura de salidas
@enduml
```

### Observaciones
- Los scripts de interfaz reutilizan la misma capa de proceso, lo que asegura paridad funcional entre ejecuciones GUI y CLI.
- La dependencia de `get_mseed_data` implica mantener credenciales y endpoints actualizados.
- `SeisBench` requiere modelos descargados previamente o acceso a internet para cargarlos al vuelo.

## 7. Modelo de datos conceptual
Aunque XAAP no persiste datos en una base relacional, los artefactos (CSV de detecciones y clasificaciones, catálogos de
configuración) pueden representarse mediante el siguiente modelo entidad-relación para facilitar la trazabilidad y una futura
normalización en base de datos:

```plantuml
@startuml
entity "Volcán" as Volcano {
  *volcan_id : string
  --
  nombre
  región
}

entity "Estación" as Station {
  *station_id : string
  --
  red
  código
  localización
}

entity "Configuración" as Config {
  *config_id : uuid
  --
  fecha_inicio
  fecha_fin
  filtros
  método_detección
}

entity "Segmento de onda" as Segment {
  *segmento_id : uuid
  --
  trace_id
  ruta_archivo
}

entity "Detección" as Detection {
  *deteccion_id : uuid
  --
  inicio
  fin
  coincidencia
  método
}

entity "Clasificación" as Classification {
  *clasificacion_id : uuid
  --
  categoría
  probabilidad
  modelo
}

entity "Modelo ML" as Model {
  *modelo_id : string
  --
  tipo
  versión
  proveedor
}

Config ||--o{ Segment : "genera"
Volcano ||--o{ Station : "monitorea"
Station ||--o{ Segment : "provee"
Segment ||--o{ Detection : "contiene"
Detection ||--o{ Classification : "produce"
Config }o--|| Volcano : "selecciona"
Classification }o--|| Model : "usa"
@enduml
```

### Implicaciones para implementación futura
- Identificadores y metadatos propuestos permiten reconstruir ejecuciones completas (qué configuración produjo cada CSV y con
  qué modelo).
- Centralizar la información en una base relacional o en un lago de datos facilitaría métricas históricas y auditoría.

## 8. Riesgos técnicos y mitigaciones
| Riesgo | Impacto | Mitigación |
| --- | --- | --- |
| Dependencia de servicios mSEED externos | Interrupciones en adquisición detienen la tubería | Implementar reintentos, cachear datos descargados, ofrecer datasets locales de respaldo |
| Tamaño de modelos SeisBench | Descargas pesadas ralentizan despliegues | Mantener repositorio interno de modelos aprobados y documentar requisitos de GPU/CPU |
| Falta de normalización en salidas CSV | Dificulta análisis comparativos | Adoptar esquema común (identificadores, unidades, huso horario) y versionar definiciones |
| Ausencia de pruebas automatizadas | Riesgo de regresiones | Priorizar harness de validación en `validate_detection` y unit tests modulares |

## 9. Suposiciones y decisiones pendientes
- Se asume disponibilidad de la librería `get_mseed_data` y credenciales válidas; de lo contrario, deben definirse simuladores.
- Parámetros de padding para detecciones profundas (`padding_start`, `padding_end`) podrían exponerse en configuración para
  mejorar control del usuario.
- La clasificación actual usa un único modelo SVM; es recomendable documentar cómo intercambiar modelos o soportar ensembles.
- La adopción de una capa de persistencia estructurada (p.ej., PostgreSQL) permanece abierta y dependerá de requisitos de
  auditoría institucional.
