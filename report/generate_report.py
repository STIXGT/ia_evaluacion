from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Image
from reportlab.lib.units import inch
import os
from datetime import datetime

def generate_pdf_report(metrics, confusion_matrix_path, output_path="output/report.pdf", detection_data=None):
    """
    Genera un reporte en PDF con los resultados detallados del modelo.
    
    :param metrics: Diccionario con métricas calculadas
    :param confusion_matrix_path: Ruta a la imagen de la matriz de confusión
    :param output_path: Ruta donde se guardará el PDF
    :param detection_data: Datos adicionales de la detección (opcional)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Función auxiliar para dibujar texto
    def draw_text(text, x, y, font="Helvetica", size=12, bold=False):
        c.setFont(f"{font}-Bold" if bold else font, size)
        c.drawString(x, y, text)
    
    # Encabezado
    draw_text("Reporte de Análisis de Detección de Vehículos", 100, height - 50, size=16, bold=True)
    draw_text(f"Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 100, height - 70)
    
    # Línea separadora
    c.line(100, height - 80, width - 100, height - 80)
    
    # Resumen ejecutivo
    y_pos = height - 110
    draw_text("Resumen Ejecutivo", 100, y_pos, bold=True)
    y_pos -= 25
    draw_text("Este reporte presenta los resultados del análisis de detección de vehículos", 100, y_pos)
    draw_text("utilizando un modelo de deep learning basado en YOLOv8.", 100, y_pos - 15)
    
    # Métricas principales
    y_pos -= 50
    draw_text("Métricas de Desempeño", 100, y_pos, bold=True)
    y_pos -= 25
    
    # Crear un cuadro para las métricas
    c.rect(100, y_pos - 60, 200, 60)
    metrics_text = [
        f"Precisión: {metrics['precision']:.3f}",
        f"Recall: {metrics['recall']:.3f}",
        f"F1-Score: {metrics['f1_score']:.3f}"
    ]
    for i, metric in enumerate(metrics_text):
        draw_text(metric, 110, y_pos - (i * 20))
    
    # Matriz de confusión
    y_pos -= 100
    if os.path.exists(confusion_matrix_path):
        draw_text("Matriz de Confusión", 100, y_pos, bold=True)
        img_width = 350
        img_height = 350
        c.drawImage(confusion_matrix_path, 100, y_pos - img_height - 10, 
                   width=img_width, height=img_height)
        y_pos -= (img_height + 30)
    
    # Análisis detallado
    draw_text("Análisis Detallado", 100, y_pos, bold=True)
    y_pos -= 25
    
    analysis_text = [
        f"• La precisión de {metrics['precision']:.3f} indica que el {metrics['precision']*100:.1f}% de las",
        "  detecciones realizadas son correctas.",
        f"• El recall de {metrics['recall']:.3f} muestra que el modelo detecta el {metrics['recall']*100:.1f}%",
        "  de los vehículos presentes en las imágenes.",
        f"• El F1-Score de {metrics['f1_score']:.3f} representa un balance entre precisión y recall,",
        "  indicando un buen desempeño general del modelo."
    ]
    
    for line in analysis_text:
        draw_text(line, 100, y_pos)
        y_pos -= 15
    
    # Recomendaciones
    y_pos -= 25
    draw_text("Recomendaciones", 100, y_pos, bold=True)
    y_pos -= 25
    
    recommendations = [
        "1. Realizar pruebas adicionales en diferentes condiciones de iluminación",
        "2. Considerar la recolección de más datos para mejorar la precisión",
        "3. Evaluar el rendimiento en tiempo real para aplicaciones en producción",
        "4. Mantener un monitoreo continuo del desempeño del modelo"
    ]
    
    for rec in recommendations:
        draw_text(rec, 100, y_pos)
        y_pos -= 15
    
    # Pie de página
    c.setFont("Helvetica", 8)
    c.drawString(100, 30, "Generado automáticamente por el sistema de análisis de detección de vehículos")
    c.drawString(width - 200, 30, f"Página 1 de 1")
    
    c.save()
    print(f"Reporte generado y guardado en: {output_path}")