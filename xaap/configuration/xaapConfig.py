
from obspy import UTCDateTime

import logging
from get_mseed_data import get_mseed_utils as gmutils
logger = logging.getLogger(__name__)

class xaapConfig():
    """
    Clase de configuración para manejar y almacenar los parámetros necesarios para la aplicación XAAP.

    Esta clase la emplean xaap_cli.py o xaap_gui.py para crear un objeto de configuración
    
    Esta clase inicializa y almacena la configuración necesaria para operar la aplicación, basándose
    en un objeto de configuración pasado durante la creación de la instancia. La configuración incluye
    parámetros para la conexión al servidor de datos mSEED, detalles de configuración de volcanes y
    estaciones, información de fechas, parámetros de filtrado y detección, y parámetros para el uso
    de modelos de deep learning.
    
    Parameters:
    - xaap_parameter: Un diccionario que contiene todos los parámetros necesarios para configurar la
                      aplicación. Debe contener claves para 'mseed', 'volcan_configuration', 'dates',
                      'filter', 'sta_lta', 'deep_learning', 'output_data', y 'xaap_directories'.
    Attributes:
    - mseed_client_id (str): ID del cliente para la conexión al servidor mSEED.
    - mseed_server_config_file (str): Ruta al archivo de configuración del servidor mSEED.
    - mseed_server_param (dict): Parámetros leídos desde el archivo de configuración del servidor mSEED.
    - volcan_volcanoes_configuration_file (str): Ruta al archivo de configuración de los volcanes.
    - volcan_station_file (str): Ruta al archivo de configuración de las estaciones.
    - volcan_volcan_name (str): Nombre del volcán para la configuración.
    - volcanoes_stations (dict): Información de las estaciones leída del archivo de configuración de volcanes.
    - stations_information (dict): Información de las estaciones leída del archivo de configuración de estaciones.
    - datetime_start (UTCDateTime): Fecha y hora de inicio para el procesamiento de datos.
    - datetime_end (UTCDateTime): Fecha y hora de fin para el procesamiento de datos.
    - filter_freq_a (float): Frecuencia A para el filtro.
    - filter_freq_b (float): Frecuencia B para el filtro.
    - filter_type (str): Tipo de filtro a aplicar.
    - sta_lta_sta (float): Valor STA para el algoritmo STA/LTA.
    - sta_lta_lta (float): Valor LTA para el algoritmo STA/LTA.
    - sta_lta_trigon (float): Valor de trigonometría para el algoritmo STA/LTA.
    - sta_lta_trigoff (float): Valor de desactivación de trigonometría para el algoritmo STA/LTA.
    - sta_lta_coincidence (int): Coincidencia requerida para el algoritmo STA/LTA.
    - sta_lta_endtime_buffer (float): Búfer de tiempo final para el algoritmo STA/LTA.
    - deep_learning_model_name (str): Nombre del modelo de deep learning.
    - deep_learning_model_version (str): Versión del modelo de deep learning.
    - deep_learning_coincidence_picks (int): Coincidencia de picks para el modelo de deep learning.
    - output_detection_folder (str): Ruta a la carpeta de salida para detecciones.
    - output_classification_folder (str): Ruta a la carpeta de salida para clasificaciones.
    - xaap_folder_comparation (str): Ruta a la carpeta de salida para comparaciones.
    - xaap_folder_config (str): Ruta a la carpeta de configuración de XAAP.
    - xaap_folder_data (str): Ruta a la carpeta de datos de XAAP.

    Raises:
    - Exception: Si hay un error al leer los archivos de configuración necesarios.
    
    El objeto creado contendrá toda la información requerida para operar la aplicación y
    estará listo para ser utilizado por las demás partes del sistema XAAP.
    
    Uso:
    >>> xaap_params = {'mseed': {'client_id': 'ARCHIVE', 'server_config_file': 'path/to/mseed_config.ini'}, ...}
    >>> config = xaapConfig(xaap_params)
    >>> print(config.mseed_client_id)
    'ARCHIVE'
    
    """


    def __init__(self,xaap_parameter):

        logger.info("start of create xaap_config object")
        
        self.mseed_client_id = xaap_parameter['mseed']['client_id']
        self.mseed_server_config_file = xaap_parameter['mseed']['server_config_file']

        try:
            self.mseed_server_param = gmutils.read_config_file(self.mseed_server_config_file)
        except Exception as e:
            raise Exception(f"Error reading mseed server config file : {e}")

        self.volcan_volcanoes_configuration_file = xaap_parameter['volcan_configuration']['volcanoes_config_file']
        self.volcan_station_file = xaap_parameter['volcan_configuration']['stations_config_file']
        self.volcan_volcan_name = xaap_parameter['volcan_configuration']['volcan_name']
        
        
        try:
            self.volcanoes_stations = gmutils.read_config_file(self.volcan_volcanoes_configuration_file)
        except Exception as e:
            raise Exception(f"Error reading volcano config file : {e}")

        try:
            self.stations_information = gmutils.read_config_file(self.volcan_station_file )

        except Exception as e:
            raise Exception(f"Error reading station config file : {e}")

        self.datetime_start = UTCDateTime(xaap_parameter["dates"]["start"])
        self.datetime_end = UTCDateTime(xaap_parameter["dates"]["end"])

        self.filter_freq_a = xaap_parameter["filter"]["freq_a"]
        self.filter_freq_b = xaap_parameter["filter"]["freq_b"]
        self.filter_type = xaap_parameter["filter"]["type"]

        self.sta_lta_sta = xaap_parameter["sta_lta"]["sta"]
        self.sta_lta_lta = xaap_parameter["sta_lta"]["lta"]
        self.sta_lta_trigon = xaap_parameter["sta_lta"]["trigon"]
        self.sta_lta_trigoff = xaap_parameter["sta_lta"]["trigoff"]
        self.sta_lta_coincidence = xaap_parameter["sta_lta"]["coincidence"]
        self.sta_lta_endtime_buffer = xaap_parameter["sta_lta"]["endtime_buffer"]

        self.deep_learning_model_name = xaap_parameter["deep_learning"]["model_name"]
        self.deep_learning_model_version = xaap_parameter["deep_learning"]["model_version"]
        self.deep_learning_coincidence_picks = int(xaap_parameter["deep_learning"]["coincidence_picks"])
        
        self.output_detection_folder = xaap_parameter["output_data"]["output_detection_folder"]
        self.output_classification_folder = xaap_parameter["output_data"]["output_classification_folder"]
        self.output_comparation_folder = xaap_parameter['output_data']["output_comparation_folder"]


        self.xaap_folder_config = xaap_parameter["xaap_directories"]["configuration_folder"]
        self.xaap_folder_data = xaap_parameter["xaap_directories"]["data_folder"]

        logger.info("xaapConfig object created")
