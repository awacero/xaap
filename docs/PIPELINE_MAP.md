# XAAP Pipeline Map (Vivo)

> Última actualización: 2026-04-10  
> Propósito: tener una vista rápida y mantenible del flujo STA/LTA y Deep Learning.

## 0) Cómo visualizar este archivo en VS Code con PlantUML

Si usas PlantUML en VS Code, copia cualquiera de los bloques `@startuml ... @enduml` de este documento a un archivo `.puml` (por ejemplo `docs/PIPELINE_MAP_HIGH_LEVEL.puml`) y abre **PlantUML: Preview Current Diagram**.

Sugerencia rápida:
1. Crea `docs/PIPELINE_MAP_HIGH_LEVEL.puml` y pega el bloque de la sección 2A.
2. Crea `docs/PIPELINE_MAP_SEQUENCE_GUI.puml` y pega el bloque de la sección 3A.
3. Usa el preview de la extensión PlantUML para exportar PNG/SVG.

---

## 1) Qué entra y qué sale

### Entradas
- Parámetros de configuración (`xaap_config`)
- Rango temporal y estaciones/canales
- Flujo sísmico (`volcan_stream`)

### Salidas principales
- Detecciones / triggers en memoria (`self.detections`, `self.triggers`)
- Visualización (plot de stream, picks y triggers)
- CSV (dependiendo de la ruta usada)

---

## 2) Mapa de alto nivel (GUI/CLI)

### 2A) PlantUML (recomendado para tu entorno)

```plantuml
@startuml
skinparam shadowing false
skinparam packageStyle rectangle

actor Operador

rectangle "GUI o CLI" as GUI
rectangle "Configuración\nxaap_config" as CFG
rectangle "request_data\nrequest_stream" as REQ
rectangle "pre_process\npre_process_stream" as PRE
diamond "Método de\ndetección" as DEC
rectangle "detect_trigger\nget_triggers" as STA
rectangle "process_deep_learning\ncreate_model" as M
rectangle "process_deep_learning\nget_detections" as D
rectangle "process_deep_learning\ncoincidence_detection" as CD
rectangle "plot_triggers" as PT
rectangle "plot_picks" as PP
rectangle "clasificación" as CL
rectangle "CSV clasificación" as CSVCL

Operador --> GUI
GUI --> CFG
CFG --> REQ
REQ --> PRE
PRE --> DEC

DEC --> STA : STA/LTA
STA --> PT

DEC --> M : Deep Learning
M --> D
D --> CD
CD --> PT
D --> PP

STA --> CL : opcional
CD --> CL : opcional
CL --> CSVCL
@enduml
```

### 2B) Mermaid (opcional)

```mermaid
flowchart TD
    A[Operador] --> B[GUI o CLI]
    B --> C[Configuración xaap_config]
    C --> D[request_data.request_stream]
    D --> E[pre_process.pre_process_stream]

    E --> F{Método de detección}

    F -->|STA/LTA| G[detect_trigger.get_triggers]
    G --> H[triggers]

    F -->|Deep Learning| I[process_deep_learning.create_model]
    I --> J[process_deep_learning.get_detections]
    J --> K[detections]
    K --> L[process_deep_learning.coincidence_detection_deep_learning]
    L --> M[coincidence triggers]

    H --> N[plot_triggers]
    M --> N
    K --> O[plot_picks]

    H -.opcional.-> P[clasificación]
    M -.opcional.-> P
    P --> Q[CSV clasificación]
```

---

## 3) Secuencia GUI (operativa)

### 3A) PlantUML (recomendado para tu entorno)

```plantuml
@startuml
skinparam shadowing false

actor Operador as U
participant xaap_gui as G
participant xaap_config as C
participant request_data as R
participant pre_process as P
participant detect_trigger as DT
participant process_deep_learning as DL

U -> G : update_parameters
G -> C : construir/actualizar configuración

U -> G : request_data
G -> R : request_stream(config)
R --> G : volcan_stream

U -> G : pre_process
G -> P : pre_process_stream(config, stream)
P --> G : stream preprocesado

alt STA/LTA
  U -> G : detection_sta_lta
  G -> DT : get_triggers(config, stream)
  DT --> G : triggers
  G -> G : plot_triggers()
else Deep Learning
  U -> G : detection_deep_learning
  G -> DL : create_model(config)
  G -> DL : get_detections(config, stream_por_estación, model)
  DL --> G : detections
  G -> DL : coincidence_detection_deep_learning(config, detections)
  DL --> G : coincidence triggers
  G -> G : plot_triggers()
  G -> G : plot_picks()
end

opt Clasificación
  U -> G : classify_detections
  G -> G : classify_detections()
  G --> U : CSV clasificación
end
@enduml
```

### 3B) Mermaid (opcional)

```mermaid
sequenceDiagram
    participant U as Operador
    participant G as xaap_gui
    participant C as xaap_config
    participant R as request_data
    participant P as pre_process
    participant DT as detect_trigger
    participant DL as process_deep_learning

    U->>G: update_parameters
    G->>C: construir/actualizar configuración

    U->>G: request_data
    G->>R: request_stream(config)
    R-->>G: volcan_stream

    U->>G: pre_process
    G->>P: pre_process_stream(config, stream)
    P-->>G: stream preprocesado

    alt STA/LTA
        U->>G: detection_sta_lta
        G->>DT: get_triggers(config, stream)
        DT-->>G: triggers
        G->>G: plot_triggers()
    else Deep Learning
        U->>G: detection_deep_learning
        G->>DL: create_model(config)
        G->>DL: get_detections(config, stream_por_estación, model)
        DL-->>G: detections
        G->>DL: coincidence_detection_deep_learning(config, detections)
        DL-->>G: coincidence triggers
        G->>G: plot_triggers()
        G->>G: plot_picks()
    end

    opt Clasificación
        U->>G: classify_detections
        G->>G: classify_detections()
        G-->>U: CSV clasificación
    end
```

---

## 4) Tabla de contratos rápidos (navegación mental)

| Paso | Función principal | Entrada | Salida | Side effects |
|---|---|---|---|---|
| Config | `configure_parameters_from_config_file` / GUI params | Archivo/árbol de parámetros | `xaap_config` | Logging |
| Data | `request_data.request_stream` | `xaap_config` | `volcan_stream` | Red/IO |
| Preproceso | `pre_process.pre_process_stream` | Config + stream | stream limpio | - |
| Detección STA/LTA | `detect_trigger.get_triggers` | Config + stream | triggers (list/dict) | Puede escribir CSV |
| Detección Deep | `process_deep_learning.get_detections` | Config + stream + model | detections normalizadas | - |
| Coincidencia Deep | `process_deep_learning.coincidence_detection_deep_learning` | Config + detections | coincidence triggers | - |
| Clasificación | `classify_detections` (GUI) / `classify_detection` (proceso) | triggers + stream | etiquetas | Escribe CSV clasificación |

---

## 5) Estado actual sobre persistencia CSV (importante)

- La ruta STA/LTA sí contempla escritura CSV en los módulos de detección tradicionales.
- La ruta Deep Learning moderna basada en `process_deep_learning` prioriza resultados en memoria (detections + coincidence) y plotting.
- Existen rutas legacy deep en `detect_trigger.py` que sí escriben CSV (útiles como referencia o migración).

---

## 6) Checklist de mantenimiento del mapa vivo

Actualiza este archivo cuando cambie alguno de estos puntos:
1. Nombre o firma de funciones del pipeline.
2. Dónde se guarda CSV (o si se elimina/agrega persistencia).
3. Orden de pasos GUI/CLI.
4. Estructura del objeto de resultados (detections/triggers).

Sugerencia: en cada PR que toque pipeline, añade una línea en el mensaje:
- `[x] PIPELINE_MAP.md actualizado`

