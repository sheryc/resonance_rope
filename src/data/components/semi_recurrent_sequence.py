from src.data.components.base_sequence_dataset import SequenceDataset


class SemiRecurrentSequenceDataset(SequenceDataset):
    def _extend_sequence(self, sequence, length):
        current_pointer = 0
        while len(sequence) < length:
            sequence.append(self.atom_arithmetic(sequence[current_pointer:(current_pointer + self.far_count)],
                                                 sequence[-self.near_count:]))
            if len(sequence) < length:
                sequence.append(
                    self.atom_arithmetic(sequence[current_pointer:(current_pointer + self.far_count)],
                                         sequence[-self.near_count:]))
                current_pointer += 1
        return sequence

    @classmethod
    def dataset_name(cls) -> str:
        return 'semirecurrent'
