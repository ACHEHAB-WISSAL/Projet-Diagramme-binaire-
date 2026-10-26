"""
Application web : équilibre liquide-vapeur eau + méthanol (flash + Rachford-Rice)

Installation :   pip install flask numpy scipy matplotlib
Lancement   :    python app.py (ouvre automatiquement la page dans le navigateur)

Composant 1 = méthanol, composant 2 = eau.
Modèles : Wilson ou idéal (Raoult). Psat par Antoine.
"""
import io
import base64
import os
import threading
import webbrowser
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from flask import Flask, request, render_template_string

app = Flask(__name__)

# =====================================================
# CONSTANTES (modifiables ici ou dans l'interface)
# =====================================================
# Antoine : log10(Psat[mmHg]) = A - B / (T[°C] + C)
ANT = {1: (8.08097, 1582.271, 239.726),    # méthanol
       2: (8.07131, 1730.630, 233.426)}    # eau
R_CAL = 1.987  # cal/(mol.K)

# Valeurs par défaut (Smith-Van Ness, table 13.10, méthanol(1)/eau(2))
DEFAULTS = dict(
    P=1.0, Tfeed=25.0, T=80.0, F=50.0, z1=0.4, model="wilson",
    V1=40.73, V2=18.07, a12=107.38, a21=469.55,       # Wilson (cal/mol, cm3/mol)
)


# =====================================================
# THERMODYNAMIQUE
# =====================================================
def psat(i, T):
    A, B, C = ANT[i]
    return 10 ** (A - B / (T + C))


def gamma(x1, T, p):
    """Coefficients d'activité (gamma1, gamma2) selon le modèle choisi."""
    if p["model"] == "ideal":
        return 1.0, 1.0
    x1 = min(max(x1, 1e-12), 1 - 1e-12)
    x2 = 1 - x1
    TK = T + 273.15
    if True:  # modèle de Wilson
        L12 = p["V2"] / p["V1"] * np.exp(-p["a12"] / (R_CAL * TK))
        L21 = p["V1"] / p["V2"] * np.exp(-p["a21"] / (R_CAL * TK))
        t = L12 / (x1 + L12 * x2) - L21 / (x2 + L21 * x1)
        ln1 = -np.log(x1 + L12 * x2) + x2 * t
        ln2 = -np.log(x2 + L21 * x1) - x1 * t
    return np.exp(ln1), np.exp(ln2)


def K_values(x1, T, p):
    g1, g2 = gamma(x1, T, p)
    Ptot = p["P"] * 760.0
    return g1 * psat(1, T) / Ptot, g2 * psat(2, T) / Ptot


# =====================================================
# RACHFORD-RICE + FLASH
# =====================================================
def rachford_rice(z1, K1, K2):
    z2 = 1 - z1
    f = lambda V: z1 * (K1 - 1) / (1 + V * (K1 - 1)) + z2 * (K2 - 1) / (1 + V * (K2 - 1))
    if f(0) <= 0:
        return 0.0      # liquide (saturé ou sous-refroidi)
    if f(1) >= 0:
        return 1.0      # vapeur (saturée ou surchauffée)
    return brentq(f, 0, 1)


def flash(z1, T, p, tol=1e-12, itmax=500):
    x1 = z1
    for _ in range(itmax):
        K1, K2 = K_values(x1, T, p)
        V = rachford_rice(z1, K1, K2)
        x_new = z1 / (1 + V * (K1 - 1))
        y1 = K1 * x_new
        if abs(x_new - x1) < tol:
            x1 = x_new
            break
        x1 = 0.5 * x1 + 0.5 * x_new
    return dict(V=V, x1=x1, y1=y1, K1=K1, K2=K2)


def T_bulle(x1, p):
    f = lambda T: x1 * K_values(x1, T, p)[0] + (1 - x1) * K_values(x1, T, p)[1] - 1
    return brentq(f, 0, 250)


def T_rosee(z1, p):
    def f(T):
        x = z1
        for _ in range(200):
            K1, K2 = K_values(x, T, p)
            a, b = z1 / K1, (1 - z1) / K2
            x_new = a / (a + b)
            if abs(x_new - x) < 1e-12:
                break
            x = 0.5 * x + 0.5 * x_new
        K1, K2 = K_values(x, T, p)
        return z1 / K1 + (1 - z1) / K2 - 1
    return brentq(f, 0, 250)


# =====================================================
# LECTURE DES PARAMETRES
# =====================================================
def read_params():
    p, errors = {}, []
    for k, d in DEFAULTS.items():
        raw = request.values.get(k)
        if raw in (None, ""):
            p[k] = d
        elif k == "model":
            p[k] = raw if raw in ("wilson", "ideal") else d
        else:
            try:
                p[k] = float(raw.replace(",", "."))
            except ValueError:
                errors.append(f"Valeur invalide pour « {k} »")
                p[k] = d
    if not 0 < p["z1"] < 1:
        errors.append("z doit être strictement entre 0 et 1.")
    if p["F"] <= 0:
        errors.append("Le débit F doit être positif.")
    if p["P"] <= 0:
        errors.append("La pression doit être positive.")
    return p, errors


# =====================================================
# GRAPHIQUES
# =====================================================
def make_diagram(p, res):
    xs = np.linspace(0, 1, 201)
    xs[0], xs[-1] = 1e-6, 1 - 1e-6
    Tb = np.array([T_bulle(x, p) for x in xs])
    ys = np.array([K_values(x, T, p)[0] * x for x, T in zip(xs, Tb)])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(xs, Tb, "b-", label="Courbe de bulle (T-x)")
    ax[0].plot(ys, Tb, "r-", label="Courbe de rosée (T-y)")
    if 0 < res["V"] < 1:
        ax[0].plot([res["x1"], res["y1"]], [p["T"], p["T"]], "ko-", label="Droite de conjugaison")
        ax[0].plot(p["z1"], p["T"], "g*", ms=15, label="Alimentation (z)")
    else:
        ax[0].plot(p["z1"], p["T"], "g*", ms=15, label="Alimentation (z, T)")
    ax[0].set(xlabel="Fraction molaire de méthanol (x, y)", ylabel="T (°C)",
              title=f"Diagramme T-x-y eau + méthanol à {p['P']:g} atm")
    ax[0].title.set_color("#e84393")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(xs, ys, "r-", label="Courbe d'équilibre")
    ax[1].plot([0, 1], [0, 1], "k--", label="y = x")
    if 0 < res["V"] < 1:
        ax[1].plot(res["x1"], res["y1"], "ko", label=f"Flash ({res['x1']:.3f} ; {res['y1']:.3f})")
    ax[1].set(xlabel="x méthanol", ylabel="y méthanol", title="Diagramme x-y",
              xlim=(0, 1), ylim=(0, 1))
    ax[1].title.set_color("#e84393")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# =====================================================
# PAGES HTML
# =====================================================
BASE = """
<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f9;color:#1c2430}
 header{background:#1f3a5f;color:#fff;padding:14px 24px}
 header h1{margin:0;font-size:1.25rem;color:#e84393}
 h2,h3{color:#e84393}
 nav{background:#2b4c7a;padding:0 24px}
 nav a{display:inline-block;color:#dce6f5;padding:10px 16px;text-decoration:none}
 nav a.on{background:#f4f6f9;color:#e84393;font-weight:600}
 main{max-width:1100px;margin:20px auto;padding:0 16px}
 .card{background:#fff;border-radius:10px;padding:18px 22px;margin-bottom:18px;box-shadow:0 1px 4px #0001}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}
 label{display:block;font-size:.85rem;color:#46566b;margin-bottom:3px}
 input,select{width:100%;box-sizing:border-box;padding:7px 9px;border:1px solid #c5cedb;border-radius:6px;font-size:.95rem}
 button{background:#1f3a5f;color:#fff;border:0;padding:10px 22px;border-radius:6px;font-size:1rem;cursor:pointer;margin-top:14px}
 button:hover{background:#2b4c7a}
 table{border-collapse:collapse;width:100%}
 th,td{padding:8px 12px;border-bottom:1px solid #e3e8ef;text-align:right}
 th:first-child,td:first-child{text-align:left}
 th{background:#eef2f8}
 .err{background:#fde8e8;color:#8a1c1c;padding:10px 14px;border-radius:6px;margin-bottom:12px}
 .info{background:#e8f1fd;color:#1b4a85;padding:10px 14px;border-radius:6px;margin:10px 0}
 img{max-width:100%}
 details summary{cursor:pointer;color:#e84393;font-weight:600;margin-top:12px}
</style></head><body>
<header><h1>Équilibre liquide-vapeur eau + méthanol — Flash &amp; Rachford-Rice</h1></header>
<nav>
 <a href="/?{{ qs }}" class="{{ 'on' if page=='calc' else '' }}">Calcul du flash</a>
 <a href="/diagramme?{{ qs }}" class="{{ 'on' if page=='diag' else '' }}">Diagramme binaire</a>
</nav>
<main>
{% for e in errors %}<div class="err">{{ e }}</div>{% endfor %}
<form method="get" action="{{ action }}" class="card">
 <div class="grid">
  <div><label>Pression P (atm)</label><input name="P" value="{{ p.P }}"></div>
  <div><label>T alimentation (°C)</label><input name="Tfeed" value="{{ p.Tfeed }}"></div>
  <div><label>T du flash (°C)</label><input name="T" value="{{ p['T'] }}"></div>
  <div><label>Débit F (mol/h)</label><input name="F" value="{{ p.F }}"></div>
  <div><label>z méthanol</label><input name="z1" value="{{ p.z1 }}"></div>
  <div><label>Modèle thermodynamique</label>
   <select name="model">
    {% for v,l in [('wilson','Wilson'),('ideal','Idéal (Raoult)')] %}
    <option value="{{ v }}" {{ 'selected' if p.model==v else '' }}>{{ l }}</option>{% endfor %}
   </select></div>
 </div>
 <details><summary>Paramètres du modèle (avancé)</summary>
  <div class="grid" style="margin-top:10px">
   <div><label>Wilson V1 (cm³/mol)</label><input name="V1" value="{{ p.V1 }}"></div>
   <div><label>Wilson V2 (cm³/mol)</label><input name="V2" value="{{ p.V2 }}"></div>
   <div><label>Wilson a12 (cal/mol)</label><input name="a12" value="{{ p.a12 }}"></div>
   <div><label>Wilson a21 (cal/mol)</label><input name="a21" value="{{ p.a21 }}"></div>
  </div>
 </details>
 <button type="submit">Calculer</button>
</form>
{% block body %}{% endblock %}
</main></body></html>
"""

CALC = BASE.replace("{% block body %}{% endblock %}", """
{% if res %}
<div class="card">
 <h2 style="margin-top:0">Résultats du flash à {{ p['T'] }} °C et {{ p.P }} atm</h2>
 <table>
  <tr><th>Grandeur</th><th>Valeur</th></tr>
  <tr><td>K méthanol</td><td>{{ '%.4f'|format(res.K1) }}</td></tr>
  <tr><td>K eau</td><td>{{ '%.4f'|format(res.K2) }}</td></tr>
  <tr><td>Fraction vaporisée V/F</td><td>{{ '%.4f'|format(res.V) }}</td></tr>
  <tr><td>Température de bulle de l'alimentation</td><td>{{ '%.2f'|format(Tb) }} °C</td></tr>
  <tr><td>Température de rosée de l'alimentation</td><td>{{ '%.2f'|format(Td) }} °C</td></tr>
 </table>
 {% if two %}
 <h3>Compositions et débits des deux phases</h3>
 <table>
  <tr><th>Phase</th><th>Débit (mol/h)</th><th>Méthanol</th><th>Eau</th></tr>
  <tr><td>Liquide (x)</td><td>{{ '%.3f'|format(Liq) }}</td><td>{{ '%.4f'|format(res.x1) }}</td><td>{{ '%.4f'|format(1-res.x1) }}</td></tr>
  <tr><td>Vapeur (y)</td><td>{{ '%.3f'|format(Vap) }}</td><td>{{ '%.4f'|format(res.y1) }}</td><td>{{ '%.4f'|format(1-res.y1) }}</td></tr>
  <tr><td>Alimentation (z)</td><td>{{ '%.3f'|format(p.F) }}</td><td>{{ '%.4f'|format(p.z1) }}</td><td>{{ '%.4f'|format(1-p.z1) }}</td></tr>
 </table>
 {% else %}
 <div class="info">À {{ p['T'] }} °C le mélange est monophasique :
  {{ 'liquide sous-refroidi (T < T bulle)' if res.V==0 else 'vapeur surchauffée (T > T rosée)' }}.
  Il n'y a pas de séparation de phases.</div>
 {% endif %}
</div>
<div class="card">
 <b>Équation résolue (Rachford-Rice) :</b> Σ zᵢ(Kᵢ − 1) / (1 + V(Kᵢ − 1)) = 0,
 puis xᵢ = zᵢ / (1 + V(Kᵢ − 1)) et yᵢ = Kᵢ·xᵢ, avec Kᵢ = γᵢ·Psatᵢ / P.
 Les γᵢ dépendent de x : le calcul est itéré jusqu'à convergence.
</div>
{% endif %}
""")

DIAG = BASE.replace("{% block body %}{% endblock %}", """
{% if img %}
<div class="card">
 <img src="data:image/png;base64,{{ img }}" alt="Diagramme binaire">
 {% if two %}
 <p>Point de flash à {{ p['T'] }} °C : <b>x = {{ '%.4f'|format(res.x1) }}</b>, <b>y = {{ '%.4f'|format(res.y1) }}</b>
  (méthanol), V/F = {{ '%.4f'|format(res.V) }}.</p>
 {% else %}
 <p class="info">Le point (z, T) est hors de la zone biphasique : pas de droite de conjugaison.</p>
 {% endif %}
</div>
{% endif %}
""")


def common_ctx(page, p, errors):
    qs = "&".join(f"{k}={v}" for k, v in p.items())
    return dict(p=p, errors=errors, page=page, qs=qs, title="Flash eau-méthanol")


@app.route("/")
def calc():
    p, errors = read_params()
    ctx = common_ctx("calc", p, errors)
    ctx["action"] = "/"
    if not errors:
        try:
            res = flash(p["z1"], p["T"], p)
            ctx.update(res=res, Tb=T_bulle(p["z1"], p), Td=T_rosee(p["z1"], p),
                       two=0 < res["V"] < 1,
                       Vap=res["V"] * p["F"], Liq=(1 - res["V"]) * p["F"])
        except Exception as e:
            ctx["errors"] = [f"Erreur de calcul : {e}"]
    return render_template_string(CALC, **ctx)


@app.route("/diagramme")
def diagramme():
    p, errors = read_params()
    ctx = common_ctx("diag", p, errors)
    ctx["action"] = "/diagramme"
    if not errors:
        try:
            res = flash(p["z1"], p["T"], p)
            ctx.update(res=res, two=0 < res["V"] < 1, img=make_diagram(p, res))
        except Exception as e:
            ctx["errors"] = [f"Erreur de calcul : {e}"]
    return render_template_string(DIAG, **ctx)


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(
            1.0,
            lambda: webbrowser.open_new("http://127.0.0.1:5000/"),
        ).start()
    app.run(debug=True)