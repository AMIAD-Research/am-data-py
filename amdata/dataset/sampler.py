
import math
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler


class CorpusLanguageBalancedSampler(Sampler):
    def __init__(
            self,
            dataset,
            corpus_ids: np.ndarray | list,
            lang_ids: np.ndarray | list,
            num_samples_per_epoch: int | None = None,
            corpus_temperature: float = 0.5,
            lang_temperature: float = 0.5,
            num_replicas: int | None = None,
            rank: int | None = None,
            seed: int = 0,
            drop_last: bool = False,
    ):
        """
        Args:
            corpus_ids: liste/array de longueur len(dataset), ex. ["fleurs", "voxlingua107", ...]
            lang_ids: liste/array de longueur len(dataset), ex. ["fra", "eng", ...]
            num_samples_per_epoch: taille totale d'une epoch (avant split DDP).
                Par défaut = len(dataset). Peut être augmenté pour voir plus souvent
                les langues rares (le tirage se fait avec remise de toute façon).
            corpus_temperature: alpha pour p(corpus) ~ count^alpha.
                1.0 = proportionnel à la taille réelle (pas d'équilibrage)
                0.0 = uniforme (équilibrage parfait entre corpus)
                0.3-0.6 = compromis usuel (cf. sampling par température multilingue)
            lang_temperature: idem, mais pour p(lang | corpus).
            num_replicas / rank: comme DistributedSampler ; auto-détectés depuis
                torch.distributed si non fournis.
            seed: seed de base, combiné à l'epoch pour le tirage déterministe partagé.
            drop_last: comme DistributedSampler.
        """
        if num_replicas is None:
            num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            )
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )

        df_corpus = dataset._corpus._implementation._table_annotation[corpus_ids]
        corpus_ids = [v for _, v in df_corpus.items()]
        corpus_ids = np.asarray(corpus_ids)

        df_lang = dataset._corpus._implementation._table_annotation[lang_ids]
        lang_ids = [v for _, v in df_lang.items()]
        lang_ids = np.asarray(lang_ids)

        assert len(corpus_ids) == len(lang_ids), "corpus_ids et lang_ids doivent être alignés"
        assert 0 <= rank < num_replicas

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.corpus_temperature = corpus_temperature
        self.lang_temperature = lang_temperature


        self.n = len(corpus_ids)

        # buckets[corpus][lang] -> np.array des indices dataset correspondants
        raw_buckets = defaultdict(lambda: defaultdict(list))
        for idx, (c, l) in enumerate(zip(corpus_ids, lang_ids)):
            raw_buckets[c][l].append(idx)
        self.buckets = {
            c: {l: np.asarray(idxs, dtype=np.int64) for l, idxs in langs.items()}
            for c, langs in raw_buckets.items()
        }
        self.corpora = sorted(self.buckets.keys())
        if len(self.corpora) == 0:
            raise ValueError("Aucun corpus trouvé : corpus_ids est vide.")

        # Probas de corpus lissées
        corpus_counts = np.array(
            [sum(len(v) for v in self.buckets[c].values()) for c in self.corpora],
            dtype=np.float64,
        )
        self.corpus_probs = self._smooth(corpus_counts, self.corpus_temperature)

        # Probas de langue lissées, calculées séparément DANS chaque corpus
        self.lang_lists = {}
        self.lang_probs = {}
        for c in self.corpora:
            langs = sorted(self.buckets[c].keys())
            counts = np.array([len(self.buckets[c][l]) for l in langs], dtype=np.float64)
            self.lang_lists[c] = langs
            self.lang_probs[c] = self._smooth(counts, self.lang_temperature)

        self.num_samples_per_epoch = num_samples_per_epoch or self.n

        if self.drop_last:
            self.num_samples = self.num_samples_per_epoch // self.num_replicas
        else:
            self.num_samples = math.ceil(self.num_samples_per_epoch / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    @staticmethod
    def _smooth(counts: np.ndarray, temperature: float) -> np.ndarray:
        if temperature == 1.0:
            return counts / counts.sum()
        weights = counts ** temperature
        return weights / weights.sum()

    def set_epoch(self, epoch: int):
        """A appeler en début de chaque epoch (comme DistributedSampler)."""
        self.epoch = epoch

    def __iter__(self):
        # RNG déterministe partagé entre tous les ranks : chaque rank calcule la
        # MEME séquence complète d'indices, puis ne garde que sa tranche -> pas de
        # doublon ni de trou entre ranks, sans communication réseau nécessaire.
        rng = np.random.default_rng(self.seed + self.epoch)

        # 1) tirage vectorisé du corpus pour chaque position
        corpus_draws = rng.choice(len(self.corpora), size=self.total_size, p=self.corpus_probs)

        indices = np.empty(self.total_size, dtype=np.int64)
        for c_idx, corpus in enumerate(self.corpora):
            mask = corpus_draws == c_idx
            n_c = int(mask.sum())
            if n_c == 0:
                continue

            lang_list = self.lang_lists[corpus]
            lang_probs = self.lang_probs[corpus]
            buckets = self.buckets[corpus]

            # 2) tirage vectorisé de la langue pour tous les échantillons de ce corpus
            lang_draws = rng.choice(len(lang_list), size=n_c, p=lang_probs)

            sub_indices = np.empty(n_c, dtype=np.int64)
            for l_idx, lang in enumerate(lang_list):
                lmask = lang_draws == l_idx
                n_l = int(lmask.sum())
                if n_l == 0:
                    continue
                bucket = buckets[lang]
                # 3) tirage uniforme (avec remise) d'exemples dans le bucket (corpus, lang)
                picks = bucket[rng.integers(0, len(bucket), size=n_l)]
                sub_indices[lmask] = picks

            indices[mask] = sub_indices

        # Split DDP : chaque rank prend une tranche contiguë disjointe
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices[start:end].tolist())

    def __len__(self):
        return self.num_samples



class DomainBalancedBatchSampler(Sampler):
    """
    BatchSampler générant des minibatches avec équilibrage HIERARCHIQUE :
      1) domain : équilibrage STRICT (quotas entiers déterministes par batch)
      2) lang   : équilibrage probabiliste (tirage pondéré, lissé par température)
      3) corpus : équilibrage probabiliste (tirage pondéré, lissé par température)

    A utiliser avec DataLoader(dataset, batch_sampler=...) -- ne PAS passer
    batch_size / shuffle / sampler au DataLoader dans ce cas.
    """

    def __init__(
            self,
            dataset,
            domain_ids: np.ndarray | list,
            lang_ids: np.ndarray | list,
            corpus_ids: np.ndarray | list,
            batch_size: int,
            num_batches_per_epoch: int | None = None,
            num_samples_per_epoch: int | None = None,
            domain_temperature: float = 0.0,
            lang_temperature: float = 0.5,
            corpus_temperature: float = 0.5,
            domain_weights: dict | None = None,
            num_replicas: int | None = None,
            rank: int | None = None,
            seed: int = 0,
            drop_last: bool = True,
    ):
        """
        Args:
            domain_ids/lang_ids/corpus_ids: labels alignés avec dataset (ex. "clean"/"noisy",
                "fra"/"eng", "fleurs"/"voxlingua107", ...)
            batch_size: taille du batch LOCAL (par rank), c'est ce que verra chaque DataLoader.
            num_batches_per_epoch: nb de batches par epoch et par rank. Prioritaire sur
                num_samples_per_epoch si fourni.
            num_samples_per_epoch: taille totale (tous ranks confondus) approximative d'une
                epoch, convertie en nb de batches. Par défaut = len(dataset).
            domain_temperature: contrôle les QUOTAS entiers par domaine dans chaque batch.
                0.0 = quotas égaux entre domaines (équilibrage strict pur).
                1.0 = quotas ~ proportionnels aux tailles réelles des domaines.
            domain_weights: si fourni, écrase domain_temperature -- dict {domain: poids}
                pour fixer manuellement la répartition du batch entre domaines
                (ex. {"clean": 0.7, "noisy": 0.3}).
            lang_temperature / corpus_temperature: comme avant, lissage probabiliste
                DANS chaque niveau parent (lang conditionné à domain, corpus conditionné
                à (domain, lang)).
            drop_last: ici, comme les batches sont toujours de taille exacte batch_size,
                ce flag n'a d'effet que sur l'arrondi du nombre de batches.
        """
        if num_replicas is None:
            num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            )
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
        assert 0 <= rank < num_replicas

        domain_ids = self._resolve(dataset, domain_ids)
        lang_ids = self._resolve(dataset, lang_ids)
        corpus_ids = self._resolve(dataset, corpus_ids)
        assert len(domain_ids) == len(lang_ids) == len(corpus_ids), \
            "domain_ids, lang_ids et corpus_ids doivent être alignés"

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.batch_size = batch_size
        self.domain_temperature = domain_temperature
        self.lang_temperature = lang_temperature
        self.corpus_temperature = corpus_temperature

        # buckets[domain][lang][corpus] -> np.array des indices dataset
        raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for idx, (d, l, c) in enumerate(zip(domain_ids, lang_ids, corpus_ids)):
            raw[d][l][c].append(idx)
        self.buckets = {
            d: {l: {c: np.asarray(v, dtype=np.int64) for c, v in corpora.items()}
                for l, corpora in langs.items()}
            for d, langs in raw.items()
        }
        self.domains = sorted(self.buckets.keys())
        if len(self.domains) == 0:
            raise ValueError("Aucun domaine trouvé : domain_ids est vide.")

        # --- Poids des domaines (utilisés pour les quotas ENTIERS par batch) ---
        if domain_weights is not None:
            w = np.array([domain_weights[d] for d in self.domains], dtype=np.float64)
            self.domain_weights = w / w.sum()
        else:
            domain_counts = np.array(
                [sum(len(v) for langs in self.buckets[d].values() for v in langs.values())
                 for d in self.domains],
                dtype=np.float64,
            )
            self.domain_weights = self._smooth(domain_counts, self.domain_temperature)

        # --- Probas de langue lissées, conditionnées à domain ---
        self.lang_lists, self.lang_probs = {}, {}
        for d in self.domains:
            langs = sorted(self.buckets[d].keys())
            counts = np.array(
                [sum(len(v) for v in self.buckets[d][l].values()) for l in langs],
                dtype=np.float64,
            )
            self.lang_lists[d] = langs
            self.lang_probs[d] = self._smooth(counts, self.lang_temperature)

        # --- Probas de corpus lissées, conditionnées à (domain, lang) ---
        self.corpus_lists, self.corpus_probs = {}, {}
        for d in self.domains:
            for l in self.lang_lists[d]:
                corpora = sorted(self.buckets[d][l].keys())
                counts = np.array([len(self.buckets[d][l][c]) for c in corpora], dtype=np.float64)
                self.corpus_lists[(d, l)] = corpora
                self.corpus_probs[(d, l)] = self._smooth(counts, self.corpus_temperature)

        # --- Nombre de batches par epoch (par rank) ---
        n = sum(len(v) for langs in self.buckets.values()
                for corpora in langs.values() for v in corpora.values())
        num_samples_per_epoch = num_samples_per_epoch or n
        if num_batches_per_epoch is not None:
            self.num_batches = num_batches_per_epoch
        else:
            samples_per_rank = num_samples_per_epoch / self.num_replicas
            if drop_last:
                self.num_batches = int(samples_per_rank // batch_size)
            else:
                self.num_batches = math.ceil(samples_per_rank / batch_size)

    @staticmethod
    def _resolve(dataset, ids: np.ndarray | list) -> np.ndarray:
        df = dataset._corpus._implementation._table_annotation[ids]
        return np.asarray([v for _, v in df.items()])

    @staticmethod
    def _smooth(counts: np.ndarray, temperature: float) -> np.ndarray:
        if temperature == 1.0:
            return counts / counts.sum()
        weights = counts ** temperature
        return weights / weights.sum()

    @staticmethod
    def _integer_quotas(total: int, weights: np.ndarray) -> np.ndarray:
        """
        Répartit `total` unités entières entre len(weights) buckets selon `weights`,
        de façon STRICTE et déterministe (méthode du plus grand reste / Hamilton).
        Garantit sum(quotas) == total exactement.
        """
        raw = weights * total
        quotas = np.floor(raw).astype(np.int64)
        remainder = total - quotas.sum()
        if remainder > 0:
            # on distribue les unités manquantes aux plus gros restes fractionnaires
            fracs = raw - quotas
            order = np.argsort(-fracs)  # tri décroissant
            quotas[order[:remainder]] += 1
        return quotas

    def set_epoch(self, epoch: int):
        """A appeler en début de chaque epoch (comme DistributedSampler)."""
        self.epoch = epoch

    def __iter__(self):
        # RNG indépendant par rank : pas de communication nécessaire, chaque rank
        # génère ses propres batches locaux (le tirage se fait avec remise de toute
        # façon, donc pas besoin de séquence globale partagée comme dans le sampler simple).
        ss = np.random.SeedSequence([self.seed, self.epoch, self.rank])
        rng = np.random.default_rng(ss)

        for _ in range(self.num_batches):
            batch = []

            # 1) quotas STRICTS par domaine pour ce batch
            domain_quotas = self._integer_quotas(self.batch_size, self.domain_weights)

            for d_idx, domain in enumerate(self.domains):
                q_d = int(domain_quotas[d_idx])
                if q_d == 0:
                    continue

                lang_list = self.lang_lists[domain]
                lang_draws = rng.choice(len(lang_list), size=q_d, p=self.lang_probs[domain])

                for l_idx, lang in enumerate(lang_list):
                    n_l = int((lang_draws == l_idx).sum())
                    if n_l == 0:
                        continue

                    corpus_list = self.corpus_lists[(domain, lang)]
                    corpus_draws = rng.choice(
                        len(corpus_list), size=n_l, p=self.corpus_probs[(domain, lang)]
                    )

                    for c_idx, corpus in enumerate(corpus_list):
                        n_c = int((corpus_draws == c_idx).sum())
                        if n_c == 0:
                            continue
                        bucket = self.buckets[domain][lang][corpus]
                        picks = bucket[rng.integers(0, len(bucket), size=n_c)]
                        batch.extend(picks.tolist())

            rng.shuffle(batch)  # évite un batch trié par domaine/lang/corpus
            yield batch

    def __len__(self):
        return self.num_batches