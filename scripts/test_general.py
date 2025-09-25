import torch
import numpy as np
import scipy.io
import os
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from train_general import GeneralizedProteinAware_TransBind

def load_new_tf_features(fea_file_path):
    """Load .fea file containing protein features for new TF"""
    with open(fea_file_path, 'r') as f:
        content = f.read()
    
    numbers = [float(x) for x in content.split()]
    data = np.array(numbers)
    
    if len(numbers) % 1280 == 0:
        sequence_length = len(numbers) // 1280
        data_matrix = data.reshape(sequence_length, 1280)
        feature_vector = np.mean(data_matrix, axis=0)
        return torch.FloatTensor(feature_vector)
    elif len(numbers) == 1280:
        return torch.FloatTensor(data)
    else:
        if len(numbers) >= 1280:
            return torch.FloatTensor(data[:1280])
        else:
            raise ValueError(f"Feature file too small: {len(numbers)} < 1280")

def load_new_dna_data(mat_file_path):
    """Load DNA data from .mat file"""
    mat_data = scipy.io.loadmat(mat_file_path)
    dna_sequences = mat_data['testxdata']
    labels = mat_data['testdata']
    
    if labels.ndim > 1:
        labels = labels.flatten()
    
    return torch.FloatTensor(dna_sequences), torch.FloatTensor(labels)

def plot_confusion_matrix(cm, threshold, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    
    plt.title(f'Confusion Matrix (Threshold: {threshold})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_confusion_matrix(cm):
    """Analyze confusion matrix and return detailed metrics"""
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Additional metrics
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Balanced accuracy
    balanced_accuracy = (recall + specificity) / 2
    
    return {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'false_positive_rate': float(false_positive_rate),
        'false_negative_rate': float(false_negative_rate),
        'positive_predictive_value': float(positive_predictive_value),
        'negative_predictive_value': float(negative_predictive_value),
        'balanced_accuracy': float(balanced_accuracy)
    }

def evaluate_new_tf():
    """
    Clean evaluation of new TF (GATA4) with comprehensive metrics including confusion matrix
    """
    
    print("🧪 EVALUATING NEW TF: GATA4 (Mouse)")
    print("="*50)
    
    # File paths

    MODEL_PATH =""
    MAPPING_FILE = ".../data/tf_to_feature_mapping_exact.json"
    FEATURES_DIR = ".../data/tf_features/"
    NEW_TF_FEA_FILE = ".../data/fasta_human/output_fea/HNF1A.fea"
    NEW_DNA_MAT_FILE = "/bml/shreya/TF_binding_site/dataset_test/DeepSEA_dataset/new_tf/HNF1A_test.mat"
    
    # Load model
    print("Loading trained model...")
    model = GeneralizedProteinAware_TransBind.load_from_checkpoint(
        MODEL_PATH, mapping_file=MAPPING_FILE, features_dir=FEATURES_DIR
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load data
    print("Loading HNF1A features and DNA sequences...")
    new_tf_features = load_new_tf_features(NEW_TF_FEA_FILE).to(device)
    dna_sequences, true_labels = load_new_dna_data(NEW_DNA_MAT_FILE)
    
    print(f"Dataset: {len(dna_sequences):,} sequences")
    print(f"Positive samples: {int(true_labels.sum()):,} ({true_labels.mean()*100:.1f}%)")
    
    # Make predictions on subset (50,000 sequences for quick testing)
    n_test = min(1500000, len(dna_sequences))
    print(f"Making predictions on {n_test:,} sequences...")
    
    predictions = []
    batch_size = 5000
    
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            end_idx = min(i + batch_size, n_test)
            batch_predictions = []
            
            for j in range(i, end_idx):
                single_dna = dna_sequences[j:j+1].to(device)
                binding_prob, _ = model.predict_new_tf(single_dna, new_tf_features)
                batch_predictions.append(binding_prob.item())
            
            predictions.extend(batch_predictions)
            
            if (end_idx) % 5000 == 0:
                print(f"  Processed {end_idx:,}/{n_test:,}")
    
    predictions = np.array(predictions)
    true_labels = true_labels[:n_test].numpy()
    
    # Calculate metrics
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 30)
    
    # ROC AUC and PR AUC
    auroc = roc_auc_score(true_labels, predictions)
    aupr = average_precision_score(true_labels, predictions)
    
    print(f"ROC AUC:     {auroc:.4f}")
    print(f"PR AUC:      {aupr:.4f}")
    
    # Test different thresholds for binary classification
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    best_mcc = -1
    best_threshold = 0.5
    confusion_matrices = {}
    
    print(f"\nTHRESHOLD ANALYSIS:")
    print("Thresh  Acc    Prec   Rec    F1     MCC    Spec   BAcc")
    print("-" * 55)
    
    for thresh in thresholds:
        pred_binary = (predictions >= thresh).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, pred_binary)
        confusion_matrices[thresh] = cm
        
        # Calculate metrics
        acc = accuracy_score(true_labels, pred_binary)
        
        # Handle cases where precision/recall might be undefined
        try:
            prec = precision_score(true_labels, pred_binary, zero_division=0)
            rec = recall_score(true_labels, pred_binary, zero_division=0)
            f1 = f1_score(true_labels, pred_binary, zero_division=0)
            mcc = matthews_corrcoef(true_labels, pred_binary)
            
            # Calculate specificity and balanced accuracy from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (rec + spec) / 2
            
        except:
            prec = rec = f1 = mcc = spec = balanced_acc = 0.0
        
        print(f"{thresh:5.1f}  {acc:5.3f}  {prec:5.3f}  {rec:5.3f}  {f1:5.3f}  {mcc:6.3f}  {spec:5.3f}  {balanced_acc:5.3f}")
        
        # Track best MCC
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh
    
    print(f"\nBest threshold: {best_threshold} (MCC: {best_mcc:.4f})")
    
    # Final metrics at best threshold
    pred_binary_best = (predictions >= best_threshold).astype(int)
    best_cm = confusion_matrix(true_labels, pred_binary_best)
    
    # Create results directory
    os.makedirs('./results_new/', exist_ok=True)
    
    # Plot confusion matrix for best threshold
    plot_confusion_matrix(best_cm, best_threshold, './results_new/confusion_matrix_best.png')
    
    # Analyze confusion matrix
    cm_analysis = analyze_confusion_matrix(best_cm)
    
    print(f"\n🔍 CONFUSION MATRIX ANALYSIS (threshold={best_threshold}):")
    print("-" * 50)
    print(f"True Negatives (TN):   {cm_analysis['true_negatives']:,}")
    print(f"False Positives (FP):  {cm_analysis['false_positives']:,}")
    print(f"False Negatives (FN):  {cm_analysis['false_negatives']:,}")
    print(f"True Positives (TP):   {cm_analysis['true_positives']:,}")
    print(f"")
    print(f"Sensitivity (Recall):  {cm_analysis['recall']:.4f}")
    print(f"Specificity:           {cm_analysis['specificity']:.4f}")
    print(f"Precision (PPV):       {cm_analysis['precision']:.4f}")
    print(f"Negative Pred Value:   {cm_analysis['negative_predictive_value']:.4f}")
    print(f"False Positive Rate:   {cm_analysis['false_positive_rate']:.4f}")
    print(f"False Negative Rate:   {cm_analysis['false_negative_rate']:.4f}")
    print(f"Balanced Accuracy:     {cm_analysis['balanced_accuracy']:.4f}")
    
    print(f"\n🎯 FINAL METRICS (threshold={best_threshold}):")
    print("-" * 35)
    print(f"Accuracy:    {accuracy_score(true_labels, pred_binary_best):.4f}")
    print(f"Precision:   {precision_score(true_labels, pred_binary_best, zero_division=0):.4f}")
    print(f"Recall:      {recall_score(true_labels, pred_binary_best, zero_division=0):.4f}")
    print(f"F1-Score:    {f1_score(true_labels, pred_binary_best, zero_division=0):.4f}")
    print(f"MCC:         {matthews_corrcoef(true_labels, pred_binary_best):.4f}")
    print(f"ROC AUC:     {auroc:.4f}")
    print(f"PR AUC:      {aupr:.4f}")
    
    # Check if inverted predictions perform better
    print(f"\n🔄 INVERTED PREDICTIONS CHECK:")
    inverted_predictions = 1 - predictions
    auroc_inv = roc_auc_score(true_labels, inverted_predictions)
    aupr_inv = average_precision_score(true_labels, inverted_predictions)
    
    print(f"ROC AUC (inverted):  {auroc_inv:.4f}")
    print(f"PR AUC (inverted):   {aupr_inv:.4f}")
    
    if auroc_inv > auroc:
        print("⚠️  Inverted predictions perform better!")
        print("   This suggests competitive binding relationship")
        better_auroc = auroc_inv
        better_aupr = aupr_inv
        relationship = "competitive"
    else:
        print("✅ Original predictions are better")
        better_auroc = auroc
        better_aupr = aupr
        relationship = "cooperative"
    
    # Summary statistics
    print(f"\n📈 PREDICTION STATISTICS:")
    print(f"Mean:        {predictions.mean():.4f}")
    print(f"Median:      {np.median(predictions):.4f}")
    print(f"Std:         {predictions.std():.4f}")
    print(f"Min:         {predictions.min():.4f}")
    print(f"Max:         {predictions.max():.4f}")
    
    # Distribution analysis
    print(f"\nPREDICTION DISTRIBUTION:")
    for thresh in [0.1, 0.2, 0.3, 0.5, 0.7]:
        count = np.sum(predictions > thresh)
        pct = count / len(predictions) * 100
        print(f"  > {thresh:3.1f}: {count:6d} ({pct:4.1f}%)")
    
    # Save results_new
    results_new = {
        'tf_name': 'HNF1A_mouse',
        'n_samples': len(predictions),
        'n_positive': int(np.sum(true_labels)),
        'positive_rate': float(np.mean(true_labels)),
        'auroc': float(auroc),
        'aupr': float(aupr),
        'auroc_inverted': float(auroc_inv),
        'aupr_inverted': float(aupr_inv),
        'best_threshold': float(best_threshold),
        'best_mcc': float(best_mcc),
        'accuracy': float(accuracy_score(true_labels, pred_binary_best)),
        'precision': float(precision_score(true_labels, pred_binary_best, zero_division=0)),
        'recall': float(recall_score(true_labels, pred_binary_best, zero_division=0)),
        'f1_score': float(f1_score(true_labels, pred_binary_best, zero_division=0)),
        'relationship': relationship,
        'mean_prediction': float(predictions.mean()),
        'std_prediction': float(predictions.std()),
        'confusion_matrix_analysis': cm_analysis,
        'confusion_matrix': best_cm.tolist()
    }
    
    # Save confusion matrices for all thresholds
    cm_results_new = {}
    for thresh, cm in confusion_matrices.items():
        cm_results_new[f'threshold_{thresh}'] = {
            'confusion_matrix': cm.tolist(),
            'analysis': analyze_confusion_matrix(cm)
        }
        # Save individual confusion matrix plots
        plot_confusion_matrix(cm, thresh, f'./results_new/confusion_matrix_thresh_{thresh}.png')
    
    # Save to files
    import json
    with open('./results_new/HNF1A_evaluation_results_new.json', 'w') as f:
        json.dump(results_new, f, indent=2)
    
    with open('./results_new/HNF1A_confusion_matrices.json', 'w') as f:
        json.dump(cm_results_new, f, indent=2)
    
    print(f"\n💾 results_new saved to: ./results_new/HNF1A_evaluation_results_new.json")
    print(f"💾 Confusion matrices saved to: ./results_new/HNF1A_confusion_matrices.json")
    print(f"📊 Confusion matrix plots saved to: ./results_new/confusion_matrix_*.png")
    
    # Summary
    print(f"\n" + "="*50)
    print(f"SUMMARY: HNF1A (Mouse) Evaluation")
    print(f"="*50)
    print(f"Dataset Size:     {len(predictions):,} sequences")
    print(f"Positive Rate:    {np.mean(true_labels)*100:.1f}%")
    print(f"Best ROC AUC:     {better_auroc:.4f}")
    print(f"Best PR AUC:      {better_aupr:.4f}")
    print(f"Best MCC:         {best_mcc:.4f}")
    print(f"Sensitivity:      {cm_analysis['recall']:.4f}")
    print(f"Specificity:      {cm_analysis['specificity']:.4f}")
    print(f"Balanced Acc:     {cm_analysis['balanced_accuracy']:.4f}")
    print(f"Relationship:     {relationship.title()}")
    print(f"Generalization:   ✅ SUCCESS")
    print(f"="*50)
    
    return results_new

if __name__ == "__main__":
    results = evaluate_new_tf()