import os
import sys
import tkinter as tk
from tkinter import filedialog

try:
    import tifffile
    import cv2
    import numpy as np
    import shapefile
except ImportError as e:
    print(f"Error al importar librerías: {e}")
    print("Asegúrese de instalar los requisitos: python -m pip install tifffile imagecodecs opencv-python numpy pyshp")
    sys.exit(1)

def read_tfw(tfw_path):
    """
    Lee un archivo de mundo de ESRI (.tfw) y devuelve los coeficientes afines.
    """
    try:
        with open(tfw_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 6:
            raise ValueError("El archivo TFW no contiene las 6 líneas requeridas.")
        
        A = float(lines[0])
        D = float(lines[1])
        B = float(lines[2])
        E = float(lines[3])
        C = float(lines[4])
        F = float(lines[5])
        
        return A, B, C, D, E, F
    except Exception as err:
        print(f"Error al leer el archivo de mundo ({tfw_path}): {err}")
        sys.exit(1)

def calcular_area_poligono(pts):
    """
    Calcula el área de un polígono usando la fórmula de Shoelace (coordenadas UTM -> metros cuadrados).
    """
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def generar_qml_style(qml_path, outline_width):
    """
    Genera un archivo de estilo de QGIS (.qml) para pintar la capa de líneas en rojo (POLYLINE).
    outline_width: Ancho de línea en milímetros.
    """
    qml_content = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="Symbology" version="3.0">
  <renderer-v2 type="simple" enableorderby="0" symbollevels="0" forcelabel="0">
    <symbols>
      <symbol type="line" name="0" alpha="1" clip_to_cut_out_bound_box="0">
        <layer class="SimpleLine" locked="0" enabled="1">
          <prop k="align_dash_pattern" v="0"/>
          <prop k="capstyle" v="square"/>
          <prop k="customdash" v="5;2"/>
          <prop k="customdash_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="customdash_unit" v="MM"/>
          <prop k="dash_pattern_on_tangent" v="0"/>
          <prop k="draw_inside_polygon" v="0"/>
          <prop k="joinstyle" v="bevel"/>
          <prop k="line_color" v="255,0,0,255"/>
          <prop k="line_style" v="solid"/>
          <prop k="line_width" v="{outline_width}"/>
          <prop k="line_width_unit" v="MM"/>
          <prop k="offset" v="0"/>
          <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="offset_unit" v="MM"/>
          <prop k="ring_filter" v="0"/>
          <prop k="trim_distance_end" v="0"/>
          <prop k="trim_distance_end_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="trim_distance_end_unit" v="MM"/>
          <prop k="trim_distance_start" v="0"/>
          <prop k="trim_distance_start_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="trim_distance_start_unit" v="MM"/>
          <prop k="tweak_dash_pattern_on_corners" v="0"/>
          <prop k="use_custom_dash" v="0"/>
          <prop k="width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""
    try:
        with open(qml_path, "w", encoding="utf-8") as f:
            f.write(qml_content)
        print(f"Archivo de estilo de QGIS (.qml) creado con éxito: {os.path.basename(qml_path)}")
    except Exception as e:
        print(f"Advertencia: No se pudo crear el archivo de estilo QML ({qml_path}): {e}")

def main():
    # 1. Configurar Tkinter y ventana de diálogo
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("Abriendo diálogo para seleccionar el archivo GeoTIFF...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    initial_dir = os.path.join(script_dir, "Geotiff")
    if not os.path.exists(initial_dir):
        initial_dir = script_dir

    file_path = filedialog.askopenfilename(
        title="Seleccionar Imagen GeoTIFF",
        filetypes=[("Archivos GeoTIFF / TIFF", "*.tif *.tiff"), ("Todos los archivos", "*.*")],
        initialdir=initial_dir
    )

    if not file_path:
        print("Operación cancelada. No se seleccionó ningún archivo.")
        sys.exit(0)

    print(f"\nArchivo GeoTIFF seleccionado: {file_path}")

    # 2. Buscar archivos auxiliares (.tfw y .prj)
    base_path, ext = os.path.splitext(file_path)
    tfw_path = base_path + ".tfw"
    if not os.path.exists(tfw_path):
        tfw_path = base_path + ".TFW"

    if not os.path.exists(tfw_path):
        print(f"Error: No se encontró el archivo de mundo georreferenciado (.tfw) en: {tfw_path}")
        sys.exit(1)

    A, B, C, D, E, F = read_tfw(tfw_path)

    # 3. Cargar la imagen usando tifffile para conservar el tipo de datos nativo (como uint16)
    print("\nCargando imagen GeoTIFF en formato nativo...")
    try:
        img = tifffile.imread(file_path)
    except Exception as e:
        print(f"Error al leer la imagen con tifffile: {e}")
        sys.exit(1)

    shape = img.shape
    print(f"Imagen cargada con éxito. Dimensiones: {shape}, Tipo de dato: {img.dtype}")

    # Determinar canales y aplicar lógica de enmascaramiento de fondo
    channels = 1
    if len(shape) == 3:
        channels = shape[2]

    mask_8 = None
    gray = None

    if channels == 1:
        gray = img
        # No hay máscara de validez para monobanda por defecto (todos válidos)
    elif channels == 2:
        # Caso típico en CAM2: Canal 0 es datos, Canal 1 es máscara (0 = fondo, 65535 = útil)
        print("\nImagen de 2 canales detectada (típica en CAM2):")
        print("  Canal 0: Banda de datos (usada para el procesamiento)")
        print("  Canal 1: Banda de máscara/validez (usada para ocultar el fondo No-Data)")
        gray = img[:, :, 0]
        
        # Generar máscara 8-bit optimizada en C++
        _, mask_raw = cv2.threshold(img[:, :, 1], 0, 255, cv2.THRESH_BINARY)
        mask_8 = mask_raw.astype(np.uint8)
    elif channels == 3:
        # Imagen RGB común: el fondo suele ser negro puro (0,0,0)
        print("\nImagen de 3 canales (RGB) detectada:")
        # Generamos la máscara combinando los 3 canales
        _, m0 = cv2.threshold(img[:, :, 0], 0, 255, cv2.THRESH_BINARY)
        _, m1 = cv2.threshold(img[:, :, 1], 0, 255, cv2.THRESH_BINARY)
        _, m2 = cv2.threshold(img[:, :, 2], 0, 255, cv2.THRESH_BINARY)
        mask_8 = cv2.bitwise_or(m0.astype(np.uint8), m1.astype(np.uint8))
        mask_8 = cv2.bitwise_or(mask_8, m2.astype(np.uint8))
        
        print("Canales de color (RGB) disponibles:")
        print("  0: Rojo (Red)")
        print("  1: Verde (Green)")
        print("  2: Azul (Blue)")
        print("  G: Convertir a escala de grises (Grayscale)")
        while True:
            choice = input("Seleccione el canal a procesar (0-2 o G): ").strip().upper()
            if choice == 'G':
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(img.dtype)
                break
            elif choice.isdigit() and 0 <= int(choice) < 3:
                gray = img[:, :, int(choice)]
                break
            else:
                print("Opción no válida. Intente de nuevo.")
    elif channels == 4:
        # Imagen RGBA común (CAM3): Canal 3 es Alfa (0 = transparente/fondo, 255 = útil)
        print("\nImagen de 4 canales (RGBA) detectada:")
        print("  El Canal 3 (Alfa) se usará automáticamente como máscara de datos válidos.")
        
        # Generar máscara 8-bit optimizada en C++
        _, mask_raw = cv2.threshold(img[:, :, 3], 0, 255, cv2.THRESH_BINARY)
        mask_8 = mask_raw.astype(np.uint8)
        
        print("Canales de color disponibles:")
        print("  0: Rojo (Red)")
        print("  1: Verde (Green)")
        print("  2: Azul (Blue)")
        print("  G: Convertir a escala de grises (Grayscale)")
        while True:
            choice = input("Seleccione el canal a procesar (0-2 o G): ").strip().upper()
            if choice == 'G':
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(img.dtype)
                break
            elif choice.isdigit() and 0 <= int(choice) < 3:
                gray = img[:, :, int(choice)]
                break
            else:
                print("Opción no válida. Intente de nuevo.")
    else:
        # Generalización
        print(f"\nImagen multibanda de {channels} canales detectada:")
        while True:
            choice = input(f"Seleccione el canal a procesar (0-{channels-1}): ").strip()
            if choice.isdigit() and 0 <= int(choice) < channels:
                gray = img[:, :, int(choice)]
                break
            else:
                print("Opción no válida.")

    # 4. Obtener estadísticas únicamente de los píxeles válidos usando cv2.minMaxLoc (bajo consumo de memoria)
    if mask_8 is not None:
        if cv2.countNonZero(mask_8) == 0:
            print("Error: Todos los píxeles están enmascarados como inválidos. No hay datos que procesar.")
            sys.exit(1)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray, mask=mask_8)
    else:
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)

    print(f"\nRango de valores en los datos útiles de la imagen: Mínimo = {min_val}, Máximo = {max_val}")

    # 5. Preguntar valor umbral en la terminal
    while True:
        threshold_str = input(f"Introduce el valor mínimo (umbral) para filtrar: ").strip()
        try:
            threshold_val = float(threshold_str)
            break
        except ValueError:
            print("Error: Por favor, ingrese un número válido.")

    if threshold_val.is_integer():
        threshold_suffix = str(int(threshold_val))
    else:
        threshold_suffix = str(threshold_val)

    # 6. Umbralizar la imagen respetando la máscara de datos válidos (evita detectar el fondo)
    print(f"\nAplicando umbral (> {threshold_val}) y enmascarando fondo...")
    _, thresh_raw = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    thresh_8 = thresh_raw.astype(np.uint8)
    
    if mask_8 is not None:
        binary_mask = cv2.bitwise_and(thresh_8, mask_8)
    else:
        binary_mask = thresh_8

    # 7. Detectar contornos vectoriales para el shape principal
    print("Detectando contornos detallados en la zona umbralizada...")
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total de contornos detallados encontrados: {len(contours)}")

    if len(contours) == 0:
        print("No se encontraron píxeles válidos que superen el umbral especificado. No se generará ningún Shapefile.")
        sys.exit(0)

    # 8. Procesar coordenadas UTM y áreas de todos los focos detallados
    contours_data = []
    for idx, contour in enumerate(contours):
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            continue

        # Proyectar vértices de píxeles a coordenadas UTM
        geo_pts = []
        for pt in pts:
            col, row = pt[0], pt[1]
            X = A * col + B * row + C
            Y = D * col + E * row + F
            geo_pts.append([X, Y])

        if geo_pts[0] != geo_pts[-1]:
            geo_pts.append(geo_pts[0])

        xs = [p[0] for p in geo_pts]
        ys = [p[1] for p in geo_pts]

        contours_data.append({
            'id': len(contours_data) + 1,
            'geo_points': geo_pts,
            'area_m2': calcular_area_poligono(geo_pts),
            'bbox': (min(xs), max(xs), min(ys), max(ys))
        })

    if len(contours_data) == 0:
        print("Ninguno de los contornos detallados detectados tiene los vértices mínimos (3) requeridos para formar un polígono.")
        sys.exit(0)

    # 9. Configurar directorio de salida en la raíz
    output_dir = os.path.join(script_dir, "Resultados_Shapefiles")
    os.makedirs(output_dir, exist_ok=True)

    file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    output_base_path = os.path.join(output_dir, f"{file_name_without_ext}_shape_{threshold_suffix}")
    shp_path = output_base_path + ".shp"

    # 10. Escribir Shapefile principal (Detallado como POLYLINE para asegurar que no se rellene)
    print(f"\nEscribiendo archivo Shapefile principal en: {shp_path}")
    w = shapefile.Writer(output_base_path, shapefile.POLYLINE)
    w.field("ID", "N", 10)
    w.field("Area_m2", "F", 18, 5)

    for item in contours_data:
        w.line([item['geo_points']])
        w.record(item['id'], item['area_m2'])
    w.close()
    print(f"¡Éxito! Shapefile principal creado con {len(contours_data)} contornos lineales detallados.")

    # Copiar .prj y crear .qml para el principal
    prj_path_in = base_path + ".prj"
    if not os.path.exists(prj_path_in):
        prj_path_in = base_path + ".PRJ"

    if os.path.exists(prj_path_in):
        try:
            with open(prj_path_in, "r", encoding="utf-8") as fin:
                prj_content = fin.read()
            with open(output_base_path + ".prj", "w", encoding="utf-8") as fout:
                fout.write(prj_content)
        except Exception as e:
            print(f"Advertencia: No se pudo copiar el archivo .prj: {e}")

    # Estilo en Rojo Fino para el detallado
    generar_qml_style(output_base_path + ".qml", outline_width=0.6)

    # 11. Detección de celdas activas por densidad (bloques de 100x100 con >= 40 píxeles por encima del umbral)
    print("\nAnalizando densidad de focos por cuadrícula de 100x100 píxeles...")
    H, W = binary_mask.shape
    cell_size = 100
    min_active_pixels = 40

    active_cells = []
    for r_idx, r in enumerate(range(0, H, cell_size)):
        r_end = min(r + cell_size, H)
        for c_idx, c in enumerate(range(0, W, cell_size)):
            c_end = min(c + cell_size, W)
            
            block = binary_mask[r:r_end, c:c_end]
            count = cv2.countNonZero(block)
            
            if count >= min_active_pixels:
                active_cells.append({
                    'r_idx': r_idx,
                    'c_idx': c_idx,
                    'r_range': (r, r_end),
                    'c_range': (c, c_end),
                    'count': count
                })

    print(f"Se encontraron {len(active_cells)} bloques de cuadrícula con alta densidad (>= {min_active_pixels} píxeles).")

    if len(active_cells) == 0:
        print("\nInformación: No se detectaron celdas de cuadrícula que cumplan con la densidad mínima (>= 40 píxeles en bloques de 100x100 píxeles). No se generará el shapefile de localización.")
        sys.exit(0)

    # 12. Agrupar celdas de alta densidad que sean adyacentes (8-conectividad)
    visited_cells = [False] * len(active_cells)
    cell_groups = []

    for i in range(len(active_cells)):
        if visited_cells[i]:
            continue
        group = [active_cells[i]]
        visited_cells[i] = True
        queue = [active_cells[i]]

        while queue:
            curr = queue.pop(0)
            for j in range(len(active_cells)):
                if not visited_cells[j]:
                    other = active_cells[j]
                    # Adyacencia horizontal, vertical o diagonal
                    if abs(curr['r_idx'] - other['r_idx']) <= 1 and abs(curr['c_idx'] - other['c_idx']) <= 1:
                        visited_cells[j] = True
                        group.append(other)
                        queue.append(other)
        cell_groups.append(group)

    print(f"Se identificaron {len(cell_groups)} zonas independientes de alta densidad.")

    # 13. Escribir Shapefile secundario (Localización/Cuadrados de Alta Densidad como POLYLINE para evitar relleno)
    output_loc_base_path = output_base_path + "_localizacion"
    shp_loc_path = output_loc_base_path + ".shp"
    print(f"Escribiendo shapefile de localización en: {shp_loc_path}")

    w_loc = shapefile.Writer(output_loc_base_path, shapefile.POLYLINE)
    w_loc.field("ID_Zona", "N", 10)
    w_loc.field("Bloques", "N", 10)
    w_loc.field("Pix_Activos", "N", 10)

    for idx, group in enumerate(cell_groups):
        # Obtener la extensión geográfica de todos los bloques en el grupo
        xs = []
        ys = []
        for cell in group:
            r_start, r_end = cell['r_range']
            c_start, c_end = cell['c_range']
            
            # Las 4 esquinas de la celda en píxeles
            corners = [
                (c_start, r_start),
                (c_end, r_start),
                (c_end, r_end),
                (c_start, r_end)
            ]
            for col, row in corners:
                X = A * col + B * row + C
                Y = D * col + E * row + F
                xs.append(X)
                ys.append(Y)

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        # Tamaño del cuadrado de localización:
        # El tamaño se ajusta al tamaño de la zona de alta densidad agrupada,
        # con un tamaño mínimo garantizado de 200m x 200m y un buffer del 20%
        width = max_x - min_x
        height = max_y - min_y
        size = max(width, height, 200.0) * 1.2

        half_size = size / 2
        sq_min_x = cx - half_size
        sq_max_x = cx + half_size
        sq_min_y = cy - half_size
        sq_max_y = cy + half_size

        # Coordenadas del cuadrado de localización
        square_coords = [
            [sq_min_x, sq_min_y],
            [sq_max_x, sq_min_y],
            [sq_max_x, sq_max_y],
            [sq_min_x, sq_max_y],
            [sq_min_x, sq_min_y]
        ]

        w_loc.line([square_coords])

        bloques = len(group)
        pix_activos = sum(cell['count'] for cell in group)
        w_loc.record(idx + 1, bloques, pix_activos)

    w_loc.close()
    print(f"¡Éxito! Shapefile de localización creado con {len(cell_groups)} zonas de concentración.")

    # Copiar .prj y crear .qml para el de localización
    if os.path.exists(prj_path_in):
        try:
            with open(prj_path_in, "r", encoding="utf-8") as fin:
                prj_content = fin.read()
            with open(output_loc_base_path + ".prj", "w", encoding="utf-8") as fout:
                fout.write(prj_content)
        except:
            pass

    # Estilo en Rojo Grueso para los cuadrados de localización
    generar_qml_style(output_loc_base_path + ".qml", outline_width=1.5)

if __name__ == "__main__":
    main()
