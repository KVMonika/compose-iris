import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {0: "malignant", 1: "benign"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
            "mean_radius": d.mean_radius,
            "mean_texture": d.mean_texture,
            "mean_perimeter": d.mean_perimeter,
            "mean_area": d.mean_area,
            "mean_smoothness": d.mean_smoothness,
            "mean_compactness": d.mean_compactness,
            "mean_concavity": d.mean_concavity,
            "mean_concave_points": d.mean_concave_points,
            "mean_symmetry": d.mean_symmetry,
            "mean_fractal_dimension": d.mean_fractal_dimension,
            "radius_error": d.radius_error,
            "texture_error": d.texture_error,
            "perimeter_error": d.perimeter_error,
            "area_error": d.area_error,
            "smoothness_error": d.smoothness_error,
            "compactness_error": d.compactness_error,
            "concavity_error": d.concavity_error,
            "concave_points_error": d.concave_points_error,
            "symmetry_error": d.symmetry_error, 
            "fractal_dimension_error": d.fractal_dimension_error,
            "worst_radius": d.worst_radius,
            "worst_texture": d.worst_texture,
            "worst_perimeter": d.worst_perimeter,
            "worst_area": d.worst_area,
            "worst_smoothness": d.worst_smoothness,
            "worst_compactness": d.worst_compactness,
            "worst_concavity": d.worst_concavity,
            "worst_concave_points": d.worst_concave_points,
            "worst_symmetry": d.worst_symmetry,
            "worst_fractal_dimension": d.worst_fractal_dimension,
            "cancer_class": d.cancer_class
        }
        for d in data
    ]

    return processed
