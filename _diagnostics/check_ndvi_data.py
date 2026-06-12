import pandas as pd

df = pd.read_csv('data_raw/ndvi_timeseries.csv')

print("=" * 80)
print("АНАЛИЗ NDVI ДАННЫХ")
print("=" * 80)
print()

print(f"📊 Всего записей: {len(df):,}")
print(f"🌾 Уникальных полей: {df['field_id'].nunique()}")
print(f"📅 Уникальных годов: {df['year'].nunique()}")
print(f"📆 Период дат: {df['date'].min()} → {df['date'].max()}")
print()

print("📈 Статистика NDVI:")
print(f"  Среднее: {df['ndvi_mean'].mean():.4f}")
print(f"  Мин: {df['ndvi_mean'].min():.4f}")
print(f"  Макс: {df['ndvi_mean'].max():.4f}")
print(f"  Медиана: {df['ndvi_mean'].median():.4f}")
print()

print("📋 Топ-10 полей по количеству наблюдений:")
field_stats = df.groupby('field_id').agg({
    'date': 'count',
    'ndvi_mean': ['mean', 'min', 'max']
}).round(4)
field_stats.columns = ['observations', 'ndvi_mean', 'ndvi_min', 'ndvi_max']
print(field_stats.sort_values('observations', ascending=False).head(10))
print()

print("📅 Распределение по годам:")
year_stats = df.groupby('year').agg({
    'field_id': 'nunique',
    'date': 'count',
    'ndvi_mean': 'mean'
}).round(4)
year_stats.columns = ['fields', 'observations', 'ndvi_avg']
print(year_stats)
print()

print("✅ Данные готовы для build_ml_dataset.py!")
