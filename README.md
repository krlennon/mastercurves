# mastercurves

Python package for automatically superimposing data sets to create a master curve, using Gaussian process regression and maximum a posteriori estimation.

Publication of this work is forthcoming. For now, if you use this software, please cite it using the metadata in the [citation](CITATION.cff) file.

## Installation

`mastercurves` is in the PyPI! You can easily install it using `pip`:

```
pip install mastercurves
```

and likewise update it:

```
pip install mastercurves --upgrade
```

## Usage

### Importing the package

Once the package has been installed, you can simply import it's modules:

```python
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply, PowerLawAge
```

### Adding data to a `MasterCurve`

To begin creating a master curve, first define a `MasterCurve` object:

```python
mc = MasterCurve()
```

and add data to the object:

```python
mc.add_data(xs, ys, states)
```

Here, `xs` is a python list (of length n) of NumPy arrays. Each of these arrays contains the x-coordinates for a data set at a particular state. Similarly, `ys` is a list of length n of NumPy arrays, each containing the y-coordinates for a data set at corresponding states. `states` is a list of length n, with elements parameterizing the different states.

### Adding coordinate transformations to the `MasterCurve`

Coordinate transforms can be added to the mastercurve as follows:

```python
mc.add_htransform(Multiply())
```

The above line tells the software to shift the data sets on the horizontal axis by a multiplicative factor. Note: by default, `Multiply()` assumes that the logarithm of the corresponding coordinate was taken before adding the data to the `MasterCurve` object. To override this, pass the argument `Multiply(scale = "linear")`. An analagous method, `add_vtransform()`, exists for transformations to the y-coordinate, which similarly assumes logarithmically scaled data.

### Superposing the data

When the data and transformations have been added to the `MasterCurve`, shifting the data is easy!

```python
mc.superpose()
```

The results can be visualized in a single line as well:

```python
mc.plot()
```

## Examples

Multiple examples of the software's use can be found in the [demos folder](demos), which demonstrate much of the functionality of the software.

## Contibuting

Inquiries and suggestions can be directed to krlennon[at]mit.edu. In particular, useful transformations can be directly added to the [transforms](mastercurves/transforms) directory, either in a local copy of the package or by raising an issue here!

## License

[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)

## References

The data used in the [demos](demos) has been generously provided by authors of the following publications:

T.  H.  Larsen  and  E.  M.  Furst,  “Microrheology  of  the liquid-solid transition during gelation,” Phys. Rev. Lett., vol. 100, p. 146001, Apr 2008.

S.  M.  Lalwani,  P.  Batys,  M.  Sammalkorpi,  and  J.  L.Lutkenhaus,  “Relaxation  Times  of  Solid-like  Polyelectrolyte Complexes of Varying pH and Water Content,” Macromolecules, vol. 54, pp. 7765–7776, Sep 2021.

The data in other examples was digitized from the following publications:

R. Gupta, B. Baldewa, and Y. M. Joshi, “Time temperature superposition in soft glassy materials,” Soft Matter, vol. 8, pp. 4171–4176, 2012.

R. I. Dekker, M. Dinkgreve, H. de Cagny, D. J. Koeze,B. P. Tighe, and D. Bonn, “Scaling of flow curves: Comparison between experiments and simulations,” Journal of Non-Newtonian Fluid Mechanics, vol. 261, pp. 33–37, 2018.

D. J. Plazek, “Temperature dependence of the viscoelastic  behavior  of  polystyrene,” The Journal of PhysicalChemistry, vol. 69, pp. 3480–3487, Oct 1965.

