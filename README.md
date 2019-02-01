# Two-Wheeled Inverted Pendulum Simulator 

This project is designed to facilitate the design of nonlinear controllers for the IP problem. More resources about this effort can be found [here](https://github.com/pgmerek/japery).

## Installation

As this project is package heavy, tools like virtualenv are recommended. With your desired python 3.5+ environment active,

```bash
git clone https://github.com/EthanJamesLew/twip_sim

cd twip_sim

pip install -r requirements.txt
```

Most scripts have simple functionality in their `__main__` only sections. For example, run

```
python twip_widget.py
```

to see a sample applet. 