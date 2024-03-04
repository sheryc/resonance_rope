from src.data.components.base_sequence_dataset import SequenceDataset


class RecurrentSequenceDataset(SequenceDataset):
    def _extend_sequence(self, sequence, length):
        for j in range(length - self.total_dependency_count):
            sequence.append(
                self.atom_arithmetic(sequence[-(self.far_count + self.near_count):-self.near_count],
                                     sequence[-self.near_count:]))
        return sequence

    @classmethod
    def dataset_name(cls) -> str:
        return 'recurrent'
