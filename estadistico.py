# conversacion: https://chat.deepseek.com/share/rckuccsj89rbjhil43

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class VideoDimensionAnalyzer:
    def __init__(self):
        self.data = {}
        self.categories = ['noViolencia', 'violencia_limpios']
        
    def parse_file(self, filename):
        """Parse el archivo de dimensiones"""
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        
        current_category = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Detectar categorías
            if line in ['noViolencia', 'violencia_limpios']:
                current_category = line
                self.data[current_category] = {}
                continue
                
            # Parsear dimensiones
            match = re.search(r'(\d+)x(\d+):\s*(\d+)\s*videos?', line)
            if match and current_category:
                width, height, count = map(int, match.groups())
                resolution = f"{width}x{height}"
                self.data[current_category][resolution] = count
                
    def calculate_resolution_metrics(self, width, height):
        """Calcula métricas para una resolución específica"""
        total_pixels = width * height
        metrics = {
            'resolution': f"{width}x{height}",
            'width': width,
            'height': height,
            'total_pixels': total_pixels,
            'aspect_ratio': round(width / height, 2) if height > 0 else 0,
            'category': self.classify_resolution(width, height)
        }
        return metrics
        
    def classify_resolution(self, width, height):
        """Clasifica la resolución en categorías"""
        total_pixels = width * height
        if total_pixels <= 480 * 360:  # < 480p
            return "Baja"
        elif total_pixels <= 1280 * 720:  # 480p-720p
            return "Media"
        else:  # > 720p
            return "Alta"
            
    def analyze_category(self, category_name):
        """Analiza una categoría específica"""
        category_data = self.data[category_name]
        total_videos = sum(category_data.values())
        
        # Calcular métricas por resolución
        resolutions = []
        for resolution, count in category_data.items():
            width, height = map(int, resolution.split('x'))
            metrics = self.calculate_resolution_metrics(width, height)
            metrics['count'] = count
            metrics['percentage'] = (count / total_videos) * 100
            resolutions.append(metrics)
            
        # Encontrar resolución más frecuente
        most_frequent = max(resolutions, key=lambda x: x['count'])
        
        # Distribución por categoría de calidad
        quality_dist = {}
        for res in resolutions:
            quality = res['category']
            if quality not in quality_dist:
                quality_dist[quality] = 0
            quality_dist[quality] += res['count']
            
        return {
            'total_videos': total_videos,
            'most_frequent': most_frequent,
            'resolutions': resolutions,
            'quality_distribution': quality_dist,
            'unique_resolutions': len(resolutions)
        }
        
    def calculate_scaling_impact(self, target_width, target_height):
        """Calcula el impacto del escalado para cada resolución"""
        target_pixels = target_width * target_height
        scaling_data = []
        
        for category in self.categories:
            for resolution, count in self.data[category].items():
                width, height = map(int, resolution.split('x'))
                original_pixels = width * height
                
                if original_pixels > target_pixels:
                    change_type = "Reducción"
                    percentage_change = ((original_pixels - target_pixels) / original_pixels) * 100
                else:
                    change_type = "Aumento"
                    percentage_change = ((target_pixels - original_pixels) / original_pixels) * 100
                    
                scaling_data.append({
                    'category': category,
                    'original_resolution': resolution,
                    'target_resolution': f"{target_width}x{target_height}",
                    'change_type': change_type,
                    'percentage_change': percentage_change,
                    'count': count,
                    'original_pixels': original_pixels,
                    'target_pixels': target_pixels
                })
                
        return scaling_data
        
    def recommend_optimal_resolution(self):
        """Recomienda la resolución óptima basada en el análisis"""
        candidate_resolutions = [
            (224, 224), (360, 360), (480, 270), 
            (640, 360), (854, 480), (1280, 720)
        ]
        
        recommendations = []
        
        for target_w, target_h in candidate_resolutions:
            scaling_data = self.calculate_scaling_impact(target_w, target_h)
            target_pixels = target_w * target_h
            
            # Calcular métricas de calidad
            avg_reduction = np.mean([x['percentage_change'] for x in scaling_data 
                                   if x['change_type'] == 'Reducción'])
            avg_increase = np.mean([x['percentage_change'] for x in scaling_data 
                                  if x['change_type'] == 'Aumento'])
            
            # Ponderar por cantidad de videos afectados
            reduction_videos = sum(x['count'] for x in scaling_data 
                                 if x['change_type'] == 'Reducción')
            increase_videos = sum(x['count'] for x in scaling_data 
                                if x['change_type'] == 'Aumento')
            
            # Score de recomendación (menor es mejor)
            score = (avg_reduction * 0.6 + avg_increase * 0.4) - (target_pixels / 100000)
            
            recommendations.append({
                'resolution': f"{target_w}x{target_h}",
                'width': target_w,
                'height': target_h,
                'pixels': target_pixels,
                'avg_reduction': avg_reduction,
                'avg_increase': avg_increase,
                'reduction_videos': reduction_videos,
                'increase_videos': increase_videos,
                'score': score,
                'aspect_ratio': round(target_w / target_h, 2)
            })
            
        return sorted(recommendations, key=lambda x: x['score'])
        
    def generate_report(self):
        """Genera el reporte completo"""
        print("=" * 60)
        print("📊 INFORME ESTADÍSTICO DE DIMENSIONES DE VIDEO")
        print("=" * 60)
        
        # Análisis por categoría
        category_analysis = {}
        for category in self.categories:
            print(f"\n📁 ANÁLISIS PARA: '{category}'")
            analysis = self.analyze_category(category)
            category_analysis[category] = analysis
            
            print(f"   📈 Total de videos: {analysis['total_videos']}")
            print(f"   🎯 Dimensión más frecuente: {analysis['most_frequent']['resolution']} "
                  f"({analysis['most_frequent']['count']} videos - {analysis['most_frequent']['percentage']:.1f}%)")
            print(f"   🔢 Resoluciones únicas: {analysis['unique_resolutions']}")
            
            print("   📊 Distribución por calidad:")
            for quality, count in analysis['quality_distribution'].items():
                percentage = (count / analysis['total_videos']) * 100
                print(f"     - {quality}: {count} videos ({percentage:.1f}%)")
                
            # Top 5 resoluciones
            top_5 = sorted(analysis['resolutions'], key=lambda x: x['count'], reverse=True)[:5]
            print("   🏆 Top 5 resoluciones:")
            for res in top_5:
                print(f"     - {res['resolution']}: {res['count']} videos ({res['percentage']:.1f}%)")
        
        # Análisis combinado
        print(f"\n🔄 ANÁLISIS COMBINADO")
        total_combined = sum(category_analysis[cat]['total_videos'] for cat in self.categories)
        print(f"   📈 Total de videos: {total_combined}")
        
        for cat in self.categories:
            percentage = (category_analysis[cat]['total_videos'] / total_combined) * 100
            print(f"   - {cat}: {category_analysis[cat]['total_videos']} videos ({percentage:.1f}%)")
        
        # Top 10 resoluciones combinadas
        all_resolutions = []
        for category in self.categories:
            for resolution, count in self.data[category].items():
                all_resolutions.append((resolution, count, category))
                
        resolution_totals = {}
        for res, count, cat in all_resolutions:
            if res not in resolution_totals:
                resolution_totals[res] = {'total': 0, 'categories': {}}
            resolution_totals[res]['total'] += count
            resolution_totals[res]['categories'][cat] = count
            
        top_10_combined = sorted(resolution_totals.items(), 
                               key=lambda x: x[1]['total'], reverse=True)[:10]
        
        print(f"\n🏆 TOP 10 RESOLUCIONES COMBINADAS:")
        for i, (res, data) in enumerate(top_10_combined, 1):
            percentage = (data['total'] / total_combined) * 100
            cat_dist = " | ".join([f"{cat}: {count}" for cat, count in data['categories'].items()])
            print(f"   {i:2d}. {res}: {data['total']} videos ({percentage:.1f}%) [{cat_dist}]")
        
        # Recomendación de resolución óptima
        print(f"\n🎯 RECOMENDACIÓN DE RESOLUCIÓN ÓPTIMA")
        recommendations = self.recommend_optimal_resolution()
        
        print("   Evaluación de candidatos:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec['resolution']} ({rec['pixels']} px, AR: {rec['aspect_ratio']})")
            print(f"      📉 Reducción promedio: {rec['avg_reduction']:.1f}% "
                  f"({rec['reduction_videos']} videos)")
            print(f"      📈 Aumento promedio: {rec['avg_increase']:.1f}% "
                  f"({rec['increase_videos']} videos)")
            print(f"      ⚖️ Score: {rec['score']:.2f}")
        
        best_rec = recommendations[0]
        print(f"\n   🏅 RECOMENDACIÓN: {best_rec['resolution']}")
        print(f"      - Balance óptimo calidad/rendimiento")
        print(f"      - Impacto promedio: {best_rec['avg_reduction']:.1f}% reducción, "
              f"{best_rec['avg_increase']:.1f}% aumento")
        print(f"      - Aspect ratio: {best_rec['aspect_ratio']} (16:9 estándar)")
        
    def create_visualizations(self):
        """Crea visualizaciones del análisis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico 1: Distribución por categoría y calidad
        categories = list(self.data.keys())
        quality_labels = ['Baja', 'Media', 'Alta']
        
        for i, category in enumerate(categories):
            analysis = self.analyze_category(category)
            qualities = [analysis['quality_distribution'].get(q, 0) for q in quality_labels]
            ax1.bar(np.array(range(len(quality_labels))) + i*0.3, qualities, 
                   width=0.3, label=category, alpha=0.8)
        
        ax1.set_title('Distribución de Videos por Calidad y Categoría')
        ax1.set_xlabel('Calidad')
        ax1.set_ylabel('Cantidad de Videos')
        ax1.set_xticks(range(len(quality_labels)))
        ax1.set_xticklabels(quality_labels)
        ax1.legend()
        
        # Gráfico 2: Top resoluciones por categoría
        for i, category in enumerate(categories):
            analysis = self.analyze_category(category)
            top_5 = sorted(analysis['resolutions'], key=lambda x: x['count'], reverse=True)[:5]
            resolutions = [f"{res['width']}x{res['height']}" for res in top_5]
            counts = [res['count'] for res in top_5]
            
            ax2.barh(np.array(range(len(resolutions))) + i*0.3, counts, 
                    height=0.3, label=category, alpha=0.8)
        
        ax2.set_title('Top 5 Resoluciones por Categoría')
        ax2.set_xlabel('Cantidad de Videos')
        ax2.set_ylabel('Resolución')
        ax2.legend()
        
        # Gráfico 3: Impacto del escalado para resolución recomendada
        best_resolution = self.recommend_optimal_resolution()[0]
        scaling_data = self.calculate_scaling_impact(best_resolution['width'], best_resolution['height'])
        
        reduction_data = [x for x in scaling_data if x['change_type'] == 'Reducción']
        increase_data = [x for x in scaling_data if x['change_type'] == 'Aumento']
        
        ax3.hist([x['percentage_change'] for x in reduction_data], 
                alpha=0.7, label='Reducción', bins=20)
        ax3.hist([x['percentage_change'] for x in increase_data], 
                alpha=0.7, label='Aumento', bins=20)
        ax3.set_title(f'Distribución del Impacto del Escalado\n(Objetivo: {best_resolution["resolution"]})')
        ax3.set_xlabel('Cambio Porcentual en Píxeles (%)')
        ax3.set_ylabel('Frecuencia')
        ax3.legend()
        
        # Gráfico 4: Comparación de resoluciones candidatas
        candidates = self.recommend_optimal_resolution()[:5]
        resolutions = [cand['resolution'] for cand in candidates]
        scores = [cand['score'] for cand in candidates]
        
        ax4.barh(resolutions, scores, color='lightgreen', alpha=0.7)
        ax4.set_title('Score de Resoluciones Candidatas\n(Menor es Mejor)')
        ax4.set_xlabel('Score de Recomendación')
        
        plt.tight_layout()
        plt.savefig('analisis_dimensiones.png', dpi=300, bbox_inches='tight')
        plt.show()

# Ejecución principal
if __name__ == "__main__":
    # Analizar datos
    analyzer = VideoDimensionAnalyzer()
    analyzer.parse_file('duracion.txt')
    
    # Generar reporte
    analyzer.generate_report()
    
    # Crear visualizaciones
    analyzer.create_visualizations()
    
    print("\n" + "=" * 60)
    print("✅ Análisis completado exitosamente!")
    print("📊 Reporte generado en consola")
    print("🖼️  Gráficos guardados como 'analisis_dimensiones.png'")
    print("=" * 60)