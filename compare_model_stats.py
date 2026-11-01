"""Print a terminal-style comparison of current CV, validation, and holdout stats."""
import json


def f(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def classification_section(label, cv_key, validation, holdout, cv):
    print(f"\n--- {label} classification ---")
    print("Model                         AUC    PR-AUC  Bal.Acc  Brier   Accuracy Precision Recall F1")
    scores = cv.get(cv_key, {}).get("scores", {})
    for name, item in scores.items():
        if name == "baselines" or "oof_metrics" not in item:
            continue
        m = item["oof_metrics"]
        print(f"{name:<28} {f(m.get('roc_auc')):>5}   {f(m.get('pr_auc')):>6}  {f(m.get('balanced_accuracy')):>7}  {f(m.get('brier_score')):>5}  {f(m.get('accuracy')):>8} {f(m.get('precision')):>9} {f(m.get('recall')):>6} {f(m.get('f1')):>5}")
    base = scores.get("baselines", {})
    if base:
        print(f"{'Always-positive baseline':<28} {f(base.get('always_positive', {}).get('roc_auc')):>5}   {f(base.get('always_positive', {}).get('pr_auc')):>6}  {'-':>7}  {'-':>5}  {f(base.get('always_positive', {}).get('accuracy')):>8}")
    print("\nValidation:", json.dumps(validation.get("classification", {}), indent=2))
    print("Frozen test:", json.dumps(holdout.get("classification", {}), indent=2))


def regression_section(label, cv_key, validation, holdout, cv):
    print(f"\n--- {label} regression ---")
    print("Model                         MAE     MedAE   RMSE     R2      Direction")
    scores = cv.get(cv_key, {}).get("scores", {})
    for name, item in scores.items():
        if name == "baselines" or "oof_metrics" not in item:
            continue
        m = item["oof_metrics"]
        print(f"{name:<28} {f(m.get('mae'), 2):>7} {f(m.get('median_absolute_error'), 2):>8} {f(m.get('rmse'), 2):>7} {f(m.get('r2')):>7} {f(m.get('directional_accuracy')):>11}")
    print("\nValidation:", json.dumps(validation.get("regression", {}), indent=2))
    print("Frozen test:", json.dumps(holdout.get("regression", {}), indent=2))


def main():
    with open("reports/model_performance_report.json", encoding="utf-8") as h:
        report = json.load(h)
    with open("reports/test_holdout_report.json", encoding="utf-8") as h:
        holdout = json.load(h)
    cv = report.get("rolling_cv", {})
    validation = report.get("validation", {})

    print("=" * 88)
    print("BULLSTREET MODEL COMPARISON")
    print("=" * 88)
    print("Selection:", report.get("selection_policy"))
    print("Test policy:", report.get("test_policy"))
    print("\nTarget coverage:")
    for horizon, details in report.get("data_quality", {}).get("coverage", {}).items():
        print(f"  {horizon}-session: {details.get('valid_return_rows')} valid return rows / {report['data_quality']['rows']} total")

    for label, cv_name, val_name, test_name in [
        ("Listing day", "listing_classification", "listing_day", "listing_day"),
        ("7 sessions", "7_classification", "7_session", "7_session"),
        ("30 sessions", "30_classification", "30_session", "30_session"),
    ]:
        classification_section(label, cv_name, validation[val_name], holdout[test_name], cv)
        regression_section(label, "listing" if label == "Listing day" else f"{label.split()[0]}_session", validation[val_name], holdout[test_name], cv)


if __name__ == "__main__":
    main()
