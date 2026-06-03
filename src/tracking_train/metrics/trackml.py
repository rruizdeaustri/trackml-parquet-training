import numpy as np
import torch
#from trackml.score import score_event
import pandas as pd
from typing import List

def _analyze_tracks(truth, submission):
    '''
    Compute the majority particle, hit counts, and weight for each track.

    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.

    Returns
    -------
    pandas.DataFrame
        Contains track_id, nhits, major_particle_id, major_particle_nhits,
        major_nhits, and major_weight columns.
    '''
    # true number of hits for each particle_id
    particles_nhits = truth['particle_id'].value_counts(sort=False)
    total_weight = truth['weight'].sum()
    # combined event with minimal reconstructed and truth information
    event = pd.merge(truth[['hit_id', 'particle_id', 'weight']],
                         submission[['hit_id', 'track_id']],
                         on=['hit_id'], how='left', validate='one_to_one')
    event.drop('hit_id', axis=1, inplace=True)
    event.sort_values(by=['track_id', 'particle_id'], inplace=True)

    # ASSUMPTIONs: 0 <= track_id, 0 <= particle_id

    tracks = []
    # running sum for the reconstructed track we are currently in
    rec_track_id = -1
    rec_nhits = 0
    # running sum for the particle we are currently in (in this track_id)
    cur_particle_id = -1
    cur_nhits = 0
    cur_weight = 0
    # majority particle with most hits up to now (in this track_id)
    maj_particle_id = -1
    maj_nhits = 0
    maj_weight = 0

    for hit in event.itertuples(index=False):
        # we reached the next track so we need to finish the current one
        if (rec_track_id != -1) and (rec_track_id != hit.track_id):
            # could be that the current particle is the majority one
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # store values for this track
            tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                particles_nhits[maj_particle_id], maj_nhits,
                maj_weight / total_weight))

        # setup running values for next track (or first)
        if rec_track_id != hit.track_id:
            rec_track_id = hit.track_id
            rec_nhits = 1
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
            maj_particle_id = -1
            maj_nhits = 0
            maj_weight= 0
            continue

        # hit is part of the current reconstructed track
        rec_nhits += 1

        # reached new particle within the same reconstructed track
        if cur_particle_id != hit.particle_id:
            # check if last particle has more hits than the majority one
            # if yes, set the last particle as the new majority particle
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # reset runnig values for current particle
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
        # hit belongs to the same particle within the same reconstructed track
        else:
            cur_nhits += 1
            cur_weight += hit.weight

    # last track is not handled inside the loop
    if maj_nhits < cur_nhits:
        maj_particle_id = cur_particle_id
        maj_nhits = cur_nhits
        maj_weight = cur_weight
    # store values for the last track
    tracks.append((rec_track_id, rec_nhits, maj_particle_id,
        particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))

    cols = ['track_id', 'nhits',
            'major_particle_id', 'major_particle_nhits',
            'major_nhits', 'major_weight']
    return pd.DataFrame.from_records(tracks, columns=cols)


def score_event(truth, submission):
    '''
    Compute the TrackML event score for a single event.

    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    '''
    tracks = _analyze_tracks(truth, submission)
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (0.5 < purity_rec) & (0.5 < purity_maj)
    return tracks['major_weight'][good_track].sum()


class MetricsCalculator:
    def __init__(self, num_classes):
        self.class_correct_counts = np.ndarray
        self.class_total_counts = np.ndarray
        self.predicted_total_counts = np.ndarray
        self.total_loss = float
        self.correct_predictions = int
        self.total_predictions = int
        self.all_true_scores = List[float]
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Resets the metrics counters. Called at the start of each epoch."""
        self.class_correct_counts = np.zeros(self.num_classes, dtype=int)
        self.class_total_counts = np.zeros(self.num_classes, dtype=int)
        self.predicted_total_counts = np.zeros(self.num_classes, dtype=int)
        self.total_loss = 0.0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.all_true_scores = []

    def update(self, outputs, labels, loss=0):
        """
        Updates values needed for metrics calculations based on the outputs of the batch,
        the correct labels, and the loss.
        """
        predicted = torch.argmax(outputs, dim=-1)

        predicted = predicted.reshape(-1)
        labels = labels.reshape(-1)

        # Ignore padding class 0
        mask = labels != 0
        predicted = predicted[mask]
        labels = labels[mask]

        if labels.numel() == 0:
            self.total_loss += loss
            return

        correct = predicted == labels

        self.correct_predictions += correct.sum().item()
        self.total_predictions += labels.numel()
        self.total_loss += loss

        label_counts = torch.bincount(
            labels,
            minlength=self.num_classes,
        ).detach().cpu().numpy()

        pred_counts = torch.bincount(
            predicted,
            minlength=self.num_classes,
        ).detach().cpu().numpy()

        correct_counts = torch.bincount(
            labels[correct],
            minlength=self.num_classes,
        ).detach().cpu().numpy()

        self.class_total_counts += label_counts
        self.predicted_total_counts += pred_counts
        self.class_correct_counts += correct_counts
        
    def update_old(self, outputs, labels, loss=0):
        """
        Updates values need for metrics calculations based on the outputs of the batch, the correct labels, and the loss.
        Called at each batch in each epoch

        Parameters:
        - outputs (Tensor): The model's outputs for the current batch.
        - labels (Tensor): The correct labels for the current batch.
        - loss : The average loss for the current batch.
        """
        _, predicted = torch.max(outputs.data, 1)  # getting predicted labels
        self.correct_predictions += (predicted == labels).sum().item()
        self.total_predictions += labels.size(0)
        self.total_loss += loss  # summing up averages losses of each batch

        # the following is truly trackml score only when the classes can resolve single tracks
        # otherwise the predicted count and label count are for all the particles belong to the class
        #   and we cannot truly resolve how many hits are predicted correctly for each particle
        # this means: no overlapping tracks in all of the events
        for c in range(
            self.num_classes
        ):  # this includes padding (class 0) but is ignored later
            # how many hits are correctly predicted for label c
            self.class_correct_counts[c] += (
                ((predicted == c) & (labels == c)).sum().item()
            )
            # how many hits have the label c
            self.class_total_counts[c] += (labels == c).sum().item()
            # how many hits are predicted to have label c
            self.predicted_total_counts[c] += (predicted == c).sum().item()

    def add_true_score(self, hit_ids, event_ids, outputs, truths_df):
        if outputs.dim() == 2:
            _, flat_predicted = torch.max(outputs, dim=1)
        elif outputs.dim() == 1:
            flat_predicted = outputs
        else:
            raise ValueError(f"add_true_score: unexpected outputs.dim()={outputs.dim()}")

        # producing matching ids and labels
        flat_hit_ids = hit_ids.view(-1)
        flat_event_ids = event_ids.view(-1)

        valid_mask = (flat_event_ids != 0) & (flat_hit_ids != 0)

        predicted = flat_predicted[valid_mask].cpu().numpy()
        hit_ids = flat_hit_ids[valid_mask].cpu().numpy()
        event_ids = flat_event_ids[valid_mask].cpu().numpy()

        predictions_df = pd.DataFrame(
            {
                "hit_id": hit_ids,
                "track_id": predicted,
                "event_id": event_ids,
            }
        )

        unique_event_ids = np.unique(event_ids)
        for event_id in unique_event_ids:
            if event_id == 0:
                continue  # skip padding
            event_predictions_df = predictions_df[
                predictions_df["event_id"] == event_id
            ]
            event_truths_df = truths_df[truths_df["event_id"] == event_id]
            
            ###--- ADDITIONAL CODE AFTER IMPLEMENTING WITH FLEX ---###
            
            # drop duplicate hit_ids that arose from overlapping windows
            if event_predictions_df['hit_id'].duplicated().any():
                print("Warning: duplicate hit_id(s) detected due to overlapping windows; dropping duplicates.")
            #    event_predictions_df = (
            #        event_predictions_df.drop_duplicates(subset='hit_id', keep='first')  #or 'last'
            #    )
                
            # ensure every hit has a non-negative, unique track_id
                event_predictions_df = (
                    event_predictions_df
                    .drop_duplicates('hit_id', keep='first')      # ← previous dedup
                    .query('track_id >= 0')                       # ← NEW: drop noise
                    .reset_index(drop=True)
                )
            
            ###----------------------------------------------------###
            
            """the function below is imported from trackml-library
            Compute the TrackML event score for a single event.
            Parameters
            ----------
            truth : pandas.DataFrame
            Truth information. Must have hit_id, particle_id, and weight columns.
            submission : pandas.DataFrame
            Proposed hit/track association. Must have hit_id and track_id columns.
            """
            event_true_score = score_event(event_truths_df, event_predictions_df)
            self.all_true_scores.append(event_true_score)

    def get_all_true_scores(self):
        return self.all_true_scores

    def calculate_accuracy(self):
        """
        Calculates the epoch-wide accuracy.
        """
        return 100 * self.correct_predictions / self.total_predictions

    def calculate_loss(self, num_batches):
        """
        Calculates the epoch-wide loss.
        """
        return (
            self.total_loss / num_batches
        )  # if the final batch is smaller, it is slightly over represented

    def calculate_trackml_score(self):
        """
        Calculates the trackml score based on double majority.
        It is the correct trackml score only when the individual tracks are resolved,
            i.e. no overlapping tracks in the all of events
        See comments in the update function.

        Returns:
        - epoch_score: The trackml score, excluding padding class.
        """
        # only calculating class success rates for non-empty true classes, excluding padding class 0
        non_zero_class_indices = np.where(
            (self.class_total_counts > 0)
            & (np.arange(len(self.class_total_counts)) != 0)
        )[0]
        # only calculating predicted success rates for non-empty predicted classes, excluding padding class 0
        non_zero_predicted_indices = np.where(
            (self.predicted_total_counts > 0)
            & (np.arange(len(self.predicted_total_counts)) != 0)
        )[0]

        # specifying float32 for the values below. This may not be necessary,
        # but it ensures that we are not using float16 which reduces precision and it not necessary,
        # since these metrics are not the primary bottleneck for memory and computation
        class_success_rates = np.zeros(self.num_classes, dtype=np.float32)
        predicted_success_rates = np.zeros(self.num_classes, dtype=np.float32)

        # Calculate rates only for non-zero label indices, i.e. for all true tracks
        # class success rate = (# hits correctly predicted for track c) / (# hits that belong to track c)
        #   i.e. % of hits in a particle that are correctly predicted
        #   if all hits produce the same label that belongs to a particle,
        #   the success rate is very low for most classes but 100% for one
        #   if each hit of a particle is put into a different bin, then the success rate is very low for all classes
        # since I ensure that the number of classes is not a small number, this is stricter than the rule that
        #   "the track should have the absolute majority of the points of the matching particle"
        #   because I cannot assume all hits produce the same label (belong to the same track)
        class_success_rates[non_zero_class_indices] = (
            self.class_correct_counts[non_zero_class_indices]
            / self.class_total_counts[non_zero_class_indices]
        )

        # predicted success rate = (# hits correctly predicted for track c) / (# hits predicted for track c)
        #   i.e. % of hits predicted for a track that correctly belong to the track
        # if all hits produce the same label that belongs to a particle, the success rate is very low for all classes
        # if only one of the hits that belong to a particle is put into that track,
        #   then the success rate is 100% for that track
        # "for a given track, the matching particle is the one to which the absolute majority of the track points belong"
        predicted_success_rates[non_zero_predicted_indices] = (
            self.class_correct_counts[non_zero_predicted_indices]
            / self.predicted_total_counts[non_zero_predicted_indices]
        )

        successful_classes_mask = (class_success_rates > 0.5) & (
            predicted_success_rates > 0.5
        )
        successful_classes_mask[0] = False  # Exclude padding class
        successful_classes = np.sum(successful_classes_mask)
        total_classes = (
            np.sum(self.class_total_counts > 0) - 1
        )  # Excluding padding class

        if total_classes == 0:
            return 0.0

        epoch_score = 100 * successful_classes / total_classes
        
        return epoch_score
