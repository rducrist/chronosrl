"""Registry of the available algorithms, mapping a name to `(get_config, algorithm class)`."""

from chronosrl.algorithms import accrl, chronosrl, crl, srl

ALGORITHMS = {
    "chronosrl": (chronosrl.get_config, chronosrl.ChronoSRL),
    "srl": (srl.get_config, srl.SRL),
    "crl": (crl.get_config, crl.CRL),
    "accrl": (accrl.get_config, accrl.ACCRL),
}
