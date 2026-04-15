//! Tests for the arrow-view-state view-state engine.

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod sort_builder_tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
    use arrow_row::SortField;
    use arrow_schema::{DataType, Field, Schema, SortOptions};

    use crate::permutation::PermutationIndex;
    use crate::sort_builder::SortBuilder;

    fn asc() -> SortOptions {
        SortOptions {
            descending: false,
            nulls_first: false,
        }
    }

    fn desc() -> SortOptions {
        SortOptions {
            descending: true,
            nulls_first: true,
        }
    }

    fn ids(idx: &PermutationIndex) -> Vec<u32> {
        idx.read_range(0, idx.len() as usize)
    }

    fn make_batch(columns: Vec<(&str, DataType, ArrayRef)>) -> RecordBatch {
        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, dt, _)| Field::new(*name, dt.clone(), true))
            .collect();
        let arrays: Vec<ArrayRef> = columns.into_iter().map(|(_, _, a)| a).collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).expect("valid batch")
    }

    // ── Single-batch sorts ─────────────────────────────────────────

    #[test]
    fn single_batch_ascending() {
        let batch = make_batch(vec![(
            "v",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![30, 10, 20])),
        )]);
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("new");
        builder.push(&batch, &[0]).expect("push");
        let idx = builder.finish().expect("finish");
        assert_eq!(ids(&idx), vec![1, 2, 0]); // 10, 20, 30
    }

    #[test]
    fn single_batch_descending() {
        let batch = make_batch(vec![(
            "v",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![30, 10, 20])),
        )]);
        let mut builder =
            SortBuilder::new(vec![SortField::new_with_options(DataType::Int64, desc())])
                .expect("new");
        builder.push(&batch, &[0]).expect("push");
        let idx = builder.finish().expect("finish");
        assert_eq!(ids(&idx), vec![0, 2, 1]); // 30, 20, 10
    }

    #[test]
    fn multi_batch() {
        let b1 = make_batch(vec![(
            "v",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![30, 10])),
        )]);
        let b2 = make_batch(vec![(
            "v",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![20, 40])),
        )]);
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("new");
        builder.push(&b1, &[0]).expect("push b1");
        builder.push(&b2, &[0]).expect("push b2");
        let idx = builder.finish().expect("finish");
        // global IDs: b1=[0,1], b2=[2,3]. Sorted: 10(1),20(2),30(0),40(3)
        assert_eq!(ids(&idx), vec![1, 2, 0, 3]);
    }

    #[test]
    fn multi_column() {
        // Sort by category (asc), then by value (desc)
        let batch = make_batch(vec![
            (
                "cat",
                DataType::Utf8,
                Arc::new(StringArray::from(vec!["B", "A", "B", "A", "C"])) as ArrayRef,
            ),
            (
                "val",
                DataType::Int64,
                Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50])) as ArrayRef,
            ),
        ]);
        let mut builder = SortBuilder::new(vec![
            SortField::new(DataType::Utf8),
            SortField::new_with_options(DataType::Int64, desc()),
        ])
        .expect("new");
        builder.push(&batch, &[0, 1]).expect("push");
        let idx = builder.finish().expect("finish");
        // A(40,row3), A(20,row1), B(30,row2), B(10,row0), C(50,row4)
        assert_eq!(ids(&idx), vec![3, 1, 2, 0, 4]);
    }

    #[test]
    fn multi_column_multi_batch() {
        let b1 = make_batch(vec![
            (
                "cat",
                DataType::Utf8,
                Arc::new(StringArray::from(vec!["B", "A"])) as ArrayRef,
            ),
            (
                "val",
                DataType::Int64,
                Arc::new(Int64Array::from(vec![10, 20])) as ArrayRef,
            ),
        ]);
        let b2 = make_batch(vec![
            (
                "cat",
                DataType::Utf8,
                Arc::new(StringArray::from(vec!["B", "A", "C"])) as ArrayRef,
            ),
            (
                "val",
                DataType::Int64,
                Arc::new(Int64Array::from(vec![30, 40, 50])) as ArrayRef,
            ),
        ]);
        let mut builder = SortBuilder::new(vec![
            SortField::new(DataType::Utf8),
            SortField::new_with_options(DataType::Int64, desc()),
        ])
        .expect("new");
        builder.push(&b1, &[0, 1]).expect("push b1");
        builder.push(&b2, &[0, 1]).expect("push b2");
        let idx = builder.finish().expect("finish");
        // global IDs: b1=[0,1], b2=[2,3,4]
        // A(40,g3), A(20,g1), B(30,g2), B(10,g0), C(50,g4)
        assert_eq!(ids(&idx), vec![3, 1, 2, 0, 4]);
    }

    #[test]
    fn strings_ascending() {
        let batch = make_batch(vec![(
            "s",
            DataType::Utf8,
            Arc::new(StringArray::from(vec!["cherry", "apple", "banana"])),
        )]);
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Utf8)]).expect("new");
        builder.push(&batch, &[0]).expect("push");
        let idx = builder.finish().expect("finish");
        assert_eq!(ids(&idx), vec![1, 2, 0]); // apple, banana, cherry
    }

    #[test]
    fn nulls_handling() {
        let batch = make_batch(vec![(
            "v",
            DataType::Int32,
            Arc::new(Int32Array::from(vec![Some(30), None, Some(10), None])),
        )]);
        let mut builder =
            SortBuilder::new(vec![SortField::new_with_options(DataType::Int32, asc())])
                .expect("new");
        builder.push(&batch, &[0]).expect("push");
        let idx = builder.finish().expect("finish");
        let result = ids(&idx);
        // ascending, nulls_first=false → non-nulls first
        assert_eq!(result[0], 2); // 10
        assert_eq!(result[1], 0); // 30
    }

    #[test]
    fn empty_batch_skipped() {
        let empty = make_batch(vec![(
            "v",
            DataType::Int64,
            Arc::new(Int64Array::from(Vec::<i64>::new())),
        )]);
        let real = make_batch(vec![(
            "v",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![20, 10])),
        )]);
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("new");
        builder.push(&empty, &[0]).expect("push empty");
        builder.push(&real, &[0]).expect("push real");
        let idx = builder.finish().expect("finish");
        assert_eq!(ids(&idx), vec![1, 0]);
    }

    #[test]
    fn empty_fails() {
        let builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("new");
        let err = builder.finish().expect_err("should fail");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn column_count_mismatch() {
        let batch = make_batch(vec![(
            "v",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![1, 2])),
        )]);
        let mut builder = SortBuilder::new(vec![
            SortField::new(DataType::Int64),
            SortField::new(DataType::Int64),
        ])
        .expect("new");
        let err = builder.push(&batch, &[0]).expect_err("should fail");
        assert!(err.to_string().contains("length mismatch"));
    }

    #[test]
    fn rows_ingested_tracking() {
        let b1 = make_batch(vec![(
            "v",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![1; 100])),
        )]);
        let b2 = make_batch(vec![(
            "v",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![2; 50])),
        )]);
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("new");
        assert_eq!(builder.rows_ingested(), 0);
        builder.push(&b1, &[0]).expect("push b1");
        assert_eq!(builder.rows_ingested(), 100);
        builder.push(&b2, &[0]).expect("push b2");
        assert_eq!(builder.rows_ingested(), 150);
    }

    #[test]
    fn float64_sort() {
        #[allow(clippy::approx_constant)]
        let batch = make_batch(vec![(
            "v",
            DataType::Float64,
            Arc::new(Float64Array::from(vec![3.14, 1.0, 2.71])),
        )]);
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Float64)]).expect("new");
        builder.push(&batch, &[0]).expect("push");
        let idx = builder.finish().expect("finish");
        assert_eq!(ids(&idx), vec![1, 2, 0]); // 1.0, 2.71, 3.14
    }

    #[test]
    fn equal_values_valid_permutation() {
        let batch = make_batch(vec![(
            "v",
            DataType::Utf8,
            Arc::new(StringArray::from(vec!["same"; 5])),
        )]);
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Utf8)]).expect("new");
        builder.push(&batch, &[0]).expect("push");
        let idx = builder.finish().expect("finish");
        let mut result = ids(&idx);
        result.sort_unstable();
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn new_empty_fields_fails() {
        let err = SortBuilder::new(vec![]).expect_err("should fail");
        assert!(err.to_string().contains("empty"));
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod natural_tests {
    use crate::permutation::PermutationIndex;

    fn ids(idx: &PermutationIndex) -> Vec<u32> {
        idx.read_range(0, idx.len() as usize)
    }

    #[test]
    fn natural_identity() {
        let idx = PermutationIndex::natural(5).expect("natural");
        assert_eq!(ids(&idx), vec![0, 1, 2, 3, 4]);
        assert_eq!(idx.len(), 5);
    }

    #[test]
    fn natural_empty() {
        let idx = PermutationIndex::natural(0).expect("natural");
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn natural_too_many_rows() {
        let result = PermutationIndex::natural(u64::from(u32::MAX) + 1);
        assert!(result.is_err());
        assert!(
            result
                .expect_err("should be TooManyRows")
                .to_string()
                .contains("exceeds u32::MAX")
        );
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod slice_tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_row::SortField;
    use arrow_schema::{DataType, Field, Schema};

    use crate::permutation::PermutationIndex;
    use crate::sort_builder::SortBuilder;

    fn ids(idx: &PermutationIndex) -> Vec<u32> {
        idx.read_range(0, idx.len() as usize)
    }

    #[test]
    fn slice_subrange() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(vec![50, 10, 30, 20, 40])) as ArrayRef],
        )
        .expect("batch");

        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("new");
        builder.push(&batch, &[0]).expect("push");
        let idx = builder.finish().expect("finish");
        // Sorted: 10(1), 20(3), 30(2), 40(4), 50(0)
        let sliced = idx.slice(2, 2);
        assert_eq!(ids(&sliced), vec![2, 4]); // 30(2), 40(4)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod filter_tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch};
    use arrow_row::SortField;
    use arrow_schema::{DataType, Field, Schema, SortOptions};

    use crate::filter::FilterIndex;
    use crate::sort_builder::SortBuilder;

    fn asc() -> SortOptions {
        SortOptions {
            descending: false,
            nulls_first: false,
        }
    }

    #[test]
    fn filter_from_ids() {
        let filter = FilterIndex::from_ids([1, 3, 5]);
        assert_eq!(filter.len(), 3);
        assert!(filter.contains(1));
        assert!(filter.contains(3));
        assert!(!filter.contains(2));
    }

    #[test]
    fn filter_from_boolean_array() {
        let arr = BooleanArray::from(vec![Some(true), Some(false), None, Some(true)]);
        let filter = FilterIndex::from_boolean_array(&arr);
        assert_eq!(filter.len(), 2);
        assert!(filter.contains(0));
        assert!(!filter.contains(1));
        assert!(!filter.contains(2));
        assert!(filter.contains(3));
    }

    #[test]
    fn filter_apply_to_permutation() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(vec![50, 10, 30, 20, 40])) as ArrayRef],
        )
        .expect("batch");

        let mut builder =
            SortBuilder::new(vec![SortField::new_with_options(DataType::Int64, asc())])
                .expect("new");
        builder.push(&batch, &[0]).expect("push");
        let sorted = builder.finish().expect("finish");
        // Sorted: [1(10), 3(20), 2(30), 4(40), 0(50)]

        let filter = FilterIndex::from_ids([0, 2, 4]);
        let filtered = filter.apply_to_permutation(&sorted);
        let ids: Vec<u32> = filtered.read_range(0, filtered.len() as usize);
        // In sort order, filtered: 2(30), 4(40), 0(50)
        assert_eq!(ids, vec![2, 4, 0]);
    }

    #[test]
    fn filter_intersection() {
        let a = FilterIndex::from_ids([1, 2, 3, 4]);
        let b = FilterIndex::from_ids([2, 4, 6]);
        let result = a.intersection(&b);
        assert_eq!(result.len(), 2);
        assert!(result.contains(2));
        assert!(result.contains(4));
    }

    #[test]
    fn filter_union() {
        let a = FilterIndex::from_ids([1, 2]);
        let b = FilterIndex::from_ids([2, 3]);
        let result = a.union(&b);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn filter_negate() {
        let filter = FilterIndex::from_ids([1, 3]);
        let negated = filter.negate(5);
        assert_eq!(negated.len(), 3);
        assert!(negated.contains(0));
        assert!(negated.contains(2));
        assert!(negated.contains(4));
        assert!(!negated.contains(1));
        assert!(!negated.contains(3));
    }

    #[test]
    fn filter_empty() {
        let filter = FilterIndex::from_ids(std::iter::empty::<u32>());
        assert!(filter.is_empty());
        assert_eq!(filter.len(), 0);
    }

    #[test]
    fn filter_iter() {
        let filter = FilterIndex::from_ids([5, 1, 3]);
        let collected: Vec<u32> = filter.iter().collect();
        assert_eq!(collected, vec![1, 3, 5]); // ascending order
    }

    #[test]
    fn filter_difference() {
        let a = FilterIndex::from_ids([0, 1, 2, 3]);
        let b = FilterIndex::from_ids([1, 3]);
        let result = a.difference(&b);
        assert_eq!(result.len(), 2);
        assert!(result.contains(0));
        assert!(result.contains(2));
        assert!(!result.contains(1));
        assert!(!result.contains(3));
    }

    #[test]
    fn filter_boolean_round_trip() {
        let original = FilterIndex::from_ids([1, 3, 5]);
        let arr = original.into_boolean_array(8);
        let restored = FilterIndex::from_boolean_array(&arr);
        assert_eq!(restored.len(), 3);
        assert!(restored.contains(1));
        assert!(restored.contains(3));
        assert!(restored.contains(5));
        assert!(!restored.contains(0));
        assert!(!restored.contains(2));
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
mod proptests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_row::SortField;
    use arrow_schema::{DataType, Field, Schema};
    use proptest::prelude::*;

    use crate::filter::FilterIndex;
    use crate::permutation::PermutationIndex;
    use crate::sort_builder::SortBuilder;

    fn build_sorted(values: &[i64]) -> PermutationIndex {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(values.to_vec())) as ArrayRef],
        )
        .expect("batch");
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("new");
        builder.push(&batch, &[0]).expect("push");
        builder.finish().expect("finish")
    }

    proptest! {
        #[test]
        fn natural_always_in_bounds(n in 1u64..100_000) {
            let idx = PermutationIndex::natural(n).expect("natural");
            let all = idx.read_range(0, idx.len() as usize);
            for &id in &all {
                prop_assert!(u64::from(id) < n);
            }
        }

        #[test]
        fn sorted_index_is_permutation(
            values in prop::collection::vec(-1000i64..1000, 1..500),
        ) {
            let n = values.len();
            let idx = build_sorted(&values);

            let mut sorted_ids = idx.read_range(0, idx.len() as usize);
            sorted_ids.sort_unstable();
            let expected: Vec<u32> = (0..n as u32).collect();
            prop_assert_eq!(sorted_ids, expected, "index must be a permutation");
        }

        #[test]
        fn sorted_values_are_monotonic(
            values in prop::collection::vec(-1000i64..1000, 1..500),
        ) {
            let n = values.len();
            let idx = build_sorted(&values);
            let perm = idx.read_range(0, idx.len() as usize);

            for i in 1..n {
                let prev = values[perm[i - 1] as usize];
                let curr = values[perm[i] as usize];
                prop_assert!(
                    curr >= prev,
                    "values should be non-decreasing: {} at virtual {} vs {} at {}",
                    prev, i - 1, curr, i
                );
            }
        }

        #[test]
        fn multi_batch_matches_single(
            values in prop::collection::vec(-1000i64..1000, 2..200),
            split in 1usize..5,
        ) {
            let single = build_sorted(&values);
            let single_ids = single.read_range(0, single.len() as usize);

            // Split into `split` batches
            let chunk_size = (values.len() / split).max(1);
            let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("new");
            let mut offset = 0;
            while offset < values.len() {
                let end = (offset + chunk_size).min(values.len());
                let chunk = &values[offset..end];
                let batch = RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)])),
                    vec![Arc::new(Int64Array::from(chunk.to_vec())) as ArrayRef],
                ).expect("batch");
                builder.push(&batch, &[0]).expect("push");
                offset = end;
            }
            let multi = builder.finish().expect("finish");
            let multi_ids = multi.read_range(0, multi.len() as usize);

            prop_assert_eq!(single_ids, multi_ids, "multi-batch must match single-batch");
        }

        #[test]
        fn filter_preserves_sort_order(
            values in prop::collection::vec(-1000i64..1000, 2..200),
            keep_fraction in 0.1f64..0.9,
        ) {
            let n = values.len();
            let sorted = build_sorted(&values);

            let keep_ids: Vec<u32> = (0..n as u32)
                .filter(|i| (*i as f64 / n as f64) < keep_fraction)
                .collect();

            if keep_ids.is_empty() {
                return Ok(());
            }

            let filter = FilterIndex::from_ids(keep_ids);
            let filtered = filter.apply_to_permutation(&sorted);

            let perm = filtered.read_range(0, filtered.len() as usize);
            for i in 1..perm.len() {
                let prev = values[perm[i - 1] as usize];
                let curr = values[perm[i] as usize];
                prop_assert!(
                    curr >= prev,
                    "filtered values should be non-decreasing"
                );
            }
        }

        #[test]
        fn filter_intersection_is_subset(
            a_ids in prop::collection::vec(0u32..1000, 0..100),
            b_ids in prop::collection::vec(0u32..1000, 0..100),
        ) {
            let a = FilterIndex::from_ids(a_ids);
            let b = FilterIndex::from_ids(b_ids);
            let intersection = a.intersection(&b);

            for id in intersection.iter() {
                prop_assert!(a.contains(id) && b.contains(id));
            }
        }

        #[test]
        fn filter_union_is_superset(
            a_ids in prop::collection::vec(0u32..1000, 0..100),
            b_ids in prop::collection::vec(0u32..1000, 0..100),
        ) {
            let a = FilterIndex::from_ids(a_ids);
            let b = FilterIndex::from_ids(b_ids);
            let union_result = a.union(&b);

            for id in a.iter() {
                prop_assert!(union_result.contains(id));
            }
            for id in b.iter() {
                prop_assert!(union_result.contains(id));
            }
        }

        #[test]
        fn filter_negate_is_complement(
            filter_ids in prop::collection::vec(0u32..100, 0..50),
        ) {
            let filter = FilterIndex::from_ids(filter_ids);
            let negated = filter.negate(100);

            for i in 0..100u32 {
                prop_assert_eq!(
                    negated.contains(i),
                    !filter.contains(i),
                    "negate must be complement at row {}", i
                );
            }
        }

        #[test]
        fn filter_difference_correct(
            a_ids in prop::collection::vec(0u32..200, 0..100),
            b_ids in prop::collection::vec(0u32..200, 0..100),
        ) {
            let a = FilterIndex::from_ids(a_ids);
            let b = FilterIndex::from_ids(b_ids);
            let diff = a.difference(&b);

            for id in diff.iter() {
                prop_assert!(a.contains(id) && !b.contains(id));
            }
            for id in a.iter() {
                if !b.contains(id) {
                    prop_assert!(diff.contains(id));
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::single_range_in_vec_init)]
mod selection_tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_row::SortField;
    use arrow_schema::{DataType, Field, Schema};

    use crate::filter::FilterIndex;
    use crate::selection::PhysicalSelection;
    use crate::sort_builder::SortBuilder;

    #[test]
    fn from_ids_contiguous() {
        let sel = PhysicalSelection::from_ids([3, 4, 5, 6]);
        assert_eq!(sel.ranges(), &[3..7]);
    }

    #[test]
    fn from_ids_scattered() {
        let sel = PhysicalSelection::from_ids([1, 3, 5, 7]);
        assert_eq!(sel.ranges(), &[1..2, 3..4, 5..6, 7..8]);
    }

    #[test]
    fn from_ids_mixed() {
        let sel = PhysicalSelection::from_ids([0, 1, 2, 5, 6, 9]);
        assert_eq!(sel.ranges(), &[0..3, 5..7, 9..10]);
    }

    #[test]
    fn from_ids_empty() {
        let sel = PhysicalSelection::from_ids(std::iter::empty::<u32>());
        assert!(sel.is_empty());
        assert_eq!(sel.row_count(), 0);
        assert_eq!(sel.ranges(), &[]);
    }

    #[test]
    fn from_ids_single() {
        let sel = PhysicalSelection::from_ids([42]);
        assert_eq!(sel.ranges(), &[42..43]);
    }

    #[test]
    fn from_ids_duplicates() {
        let sel = PhysicalSelection::from_ids([3, 3, 5, 5]);
        assert_eq!(sel.ranges(), &[3..4, 5..6]);
    }

    #[test]
    fn row_count() {
        let sel = PhysicalSelection::from_ids([0, 1, 2, 5, 6, 9]);
        assert_eq!(sel.row_count(), 6);
    }

    #[test]
    fn iter_ids() {
        let sel = PhysicalSelection::from_ids([0, 1, 2, 5, 6, 9]);
        let ids: Vec<u32> = sel.iter_ids().collect();
        assert_eq!(ids, vec![0, 1, 2, 5, 6, 9]);
    }

    #[test]
    fn permutation_to_physical_selection() {
        // Values: [50, 10, 30, 20, 40]
        // Sorted ascending: 10(1), 20(3), 30(2), 40(4), 50(0)
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(vec![50, 10, 30, 20, 40])) as ArrayRef],
        )
        .expect("batch");
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("new");
        builder.push(&batch, &[0]).expect("push");
        let idx = builder.finish().expect("finish");

        // Virtual rows 1..4 → values 20(3), 30(2), 40(4)
        let (sel, virtual_ids) = idx.to_physical_selection(1, 4);
        assert_eq!(virtual_ids, vec![3, 2, 4]);
        // Physical IDs {2,3,4} → merged range 2..5
        assert_eq!(sel.ranges(), &[2..5]);
    }

    #[test]
    fn virtual_order_map_correctness() {
        let sel = PhysicalSelection::from_ids([2, 3, 4]);
        // virtual_ids: [3, 2, 4] means virtual pos 0→phys 3, pos 1→phys 2, pos 2→phys 4
        let map = sel.virtual_order_map(&[3, 2, 4]);
        // Physical ascending: 2,3,4
        // phys 2 → virtual pos 1 → map[0] = 1
        // phys 3 → virtual pos 0 → map[1] = 0
        // phys 4 → virtual pos 2 → map[2] = 2
        assert_eq!(map, vec![1, 0, 2]);
    }

    #[test]
    fn physical_selection_with_filter() {
        // Values: [50, 10, 30, 20, 40]
        // Sorted ascending: 10(1), 20(3), 30(2), 40(4), 50(0)
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(vec![50, 10, 30, 20, 40])) as ArrayRef],
        )
        .expect("batch");
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("new");
        builder.push(&batch, &[0]).expect("push");
        let sorted = builder.finish().expect("finish");

        // Filter: keep rows 0, 2, 4 → values 50, 30, 40
        let filter = FilterIndex::from_ids([0, 2, 4]);
        let filtered = filter.apply_to_permutation(&sorted);
        // Filtered sorted: 30(2), 40(4), 50(0)

        let (sel, virtual_ids) = filtered.to_physical_selection(0, 3);
        assert_eq!(virtual_ids, vec![2, 4, 0]);
        // Physical IDs {0, 2, 4} → ranges [0..1, 2..3, 4..5]
        assert_eq!(sel.ranges(), &[0..1, 2..3, 4..5]);
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod selection_proptests {
    use proptest::prelude::*;

    use crate::selection::PhysicalSelection;

    proptest! {
        #[test]
        fn ranges_sorted_nonoverlapping(
            ids in prop::collection::vec(0u32..10_000, 0..500),
        ) {
            let sel = PhysicalSelection::from_ids(ids);
            let ranges = sel.ranges();
            for r in ranges {
                prop_assert!(r.start < r.end, "range must be non-empty");
            }
            for w in ranges.windows(2) {
                prop_assert!(
                    w[0].end <= w[1].start,
                    "ranges must be sorted and non-overlapping: {:?} vs {:?}",
                    w[0], w[1]
                );
            }
        }

        #[test]
        fn row_count_matches_unique_ids(
            ids in prop::collection::vec(0u32..10_000, 0..500),
        ) {
            let mut unique = ids.clone();
            unique.sort_unstable();
            unique.dedup();
            let sel = PhysicalSelection::from_ids(ids);
            prop_assert_eq!(sel.row_count(), unique.len());
        }

        #[test]
        fn virtual_order_map_is_valid_permutation(
            ids in prop::collection::vec(0u32..10_000, 1..200),
        ) {
            let mut unique = ids.clone();
            unique.sort_unstable();
            unique.dedup();

            if unique.is_empty() {
                return Ok(());
            }

            let sel = PhysicalSelection::from_ids(unique.iter().copied());
            // virtual_ids in some shuffled order
            let mut virtual_ids = unique.clone();
            // simple deterministic shuffle: reverse
            virtual_ids.reverse();

            let map = sel.virtual_order_map(&virtual_ids);
            prop_assert_eq!(map.len(), unique.len());

            let mut sorted_map = map.clone();
            sorted_map.sort_unstable();
            let expected: Vec<usize> = (0..unique.len()).collect();
            prop_assert_eq!(sorted_map, expected, "map must be a valid permutation");
        }
    }
}

// ─── Evaluate tests ───────────────────────────────────────────────────────────
#[cfg(feature = "evaluate")]
#[allow(clippy::unwrap_used)]
mod evaluate_tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BooleanArray, Date32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
        StringArray,
    };
    use arrow_schema::{DataType, Field, Schema};

    use crate::evaluate::{ScalarPredicate, ScalarValue, evaluate, evaluate_batches};

    #[test]
    fn eval_eq_int64() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 20, 10]));
        let f = evaluate(&col, &ScalarPredicate::Eq(ScalarValue::Int64(20))).unwrap();
        let ids: Vec<u32> = f.iter().collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn eval_neq_string() {
        let col: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "a", "c"]));
        let f = evaluate(&col, &ScalarPredicate::NotEq(ScalarValue::Utf8("a".into()))).unwrap();
        let ids: Vec<u32> = f.iter().collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn eval_lt_float64() {
        let col: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.5, 3.0, 0.5]));
        let f = evaluate(&col, &ScalarPredicate::Lt(ScalarValue::Float64(2.0))).unwrap();
        let ids: Vec<u32> = f.iter().collect();
        assert_eq!(ids, vec![0, 3]);
    }

    #[test]
    fn eval_gte_int32() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        let f = evaluate(&col, &ScalarPredicate::Gte(ScalarValue::Int32(3))).unwrap();
        let ids: Vec<u32> = f.iter().collect();
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn eval_between_int64() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![1, 5, 10, 15, 20]));
        let f = evaluate(
            &col,
            &ScalarPredicate::Between {
                lo: ScalarValue::Int64(5),
                hi: ScalarValue::Int64(15),
            },
        )
        .unwrap();
        let ids: Vec<u32> = f.iter().collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn eval_is_null() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3), None]));
        let f = evaluate(&col, &ScalarPredicate::IsNull).unwrap();
        let ids: Vec<u32> = f.iter().collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn eval_is_not_null() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3), None]));
        let f = evaluate(&col, &ScalarPredicate::IsNotNull).unwrap();
        let ids: Vec<u32> = f.iter().collect();
        assert_eq!(ids, vec![0, 2]);
    }

    #[test]
    fn eval_empty_array() {
        let col: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        let f = evaluate(&col, &ScalarPredicate::Eq(ScalarValue::Int64(5))).unwrap();
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn eval_all_nulls() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![None, None, None] as Vec<Option<i64>>));
        let f = evaluate(&col, &ScalarPredicate::Eq(ScalarValue::Int64(5))).unwrap();
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn eval_batches_multi() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let b1 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![10, 20, 30]))],
        )
        .unwrap();
        let b2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![20, 40]))],
        )
        .unwrap();

        let pred = ScalarPredicate::Eq(ScalarValue::Int64(20));
        let f = evaluate_batches(vec![(b1, 0), (b2, 0)].into_iter(), &pred).unwrap();
        let ids: Vec<u32> = f.iter().collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn eval_batches_empty_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let empty = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(Vec::<i64>::new()))],
        )
        .unwrap();
        let b2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![5, 10]))],
        )
        .unwrap();

        let pred = ScalarPredicate::Eq(ScalarValue::Int64(10));
        let f = evaluate_batches(vec![(empty, 0), (b2, 0)].into_iter(), &pred).unwrap();
        let ids: Vec<u32> = f.iter().collect();
        // empty + [5,10] → global id 1 (row 1 of batch2, offset 0 from empty)
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn eval_date32() {
        let col: ArrayRef = Arc::new(Date32Array::from(vec![17000, 18000, 19000, 20000]));
        let f = evaluate(&col, &ScalarPredicate::Gt(ScalarValue::Date32(18000))).unwrap();
        let ids: Vec<u32> = f.iter().collect();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn eval_bool_column() {
        let col: ArrayRef = Arc::new(BooleanArray::from(vec![true, false, true]));
        let f = evaluate(&col, &ScalarPredicate::Eq(ScalarValue::Bool(true))).unwrap();
        let ids: Vec<u32> = f.iter().collect();
        assert_eq!(ids, vec![0, 2]);
    }
}

#[cfg(feature = "evaluate")]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
mod evaluate_proptests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array};
    use proptest::prelude::*;

    use crate::evaluate::{ScalarPredicate, ScalarValue, evaluate};

    proptest! {
        #[test]
        fn prop_eval_eq_matches_manual(
            values in prop::collection::vec(-100i64..100, 0..200),
            target in -100i64..100,
        ) {
            let col: ArrayRef = Arc::new(Int64Array::from(values.clone()));
            let f = evaluate(&col, &ScalarPredicate::Eq(ScalarValue::Int64(target))).unwrap();
            let ids: Vec<u32> = f.iter().collect();

            let expected: Vec<u32> = values.iter().enumerate()
                .filter(|(_, v)| **v == target)
                .map(|(i, _)| i as u32)
                .collect();

            prop_assert_eq!(ids, expected);
        }

        #[test]
        fn prop_eval_between_subset(
            values in prop::collection::vec(-100i64..100, 1..200),
            a in -100i64..100,
            b in -100i64..100,
        ) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let col: ArrayRef = Arc::new(Int64Array::from(values.clone()));

            let between = evaluate(&col, &ScalarPredicate::Between {
                lo: ScalarValue::Int64(lo),
                hi: ScalarValue::Int64(hi),
            }).unwrap();

            let gte = evaluate(&col, &ScalarPredicate::Gte(ScalarValue::Int64(lo))).unwrap();
            let lte = evaluate(&col, &ScalarPredicate::Lte(ScalarValue::Int64(hi))).unwrap();
            let intersection = gte.intersection(&lte);

            let between_ids: Vec<u32> = between.iter().collect();
            let inter_ids: Vec<u32> = intersection.iter().collect();
            prop_assert_eq!(between_ids, inter_ids);
        }

        #[test]
        fn prop_eval_null_covers_all(
            n_nulls in 0usize..50,
            n_valid in 0usize..50,
        ) {
            let total = n_nulls + n_valid;
            let mut values: Vec<Option<i64>> = Vec::with_capacity(total);
            values.extend((0..n_valid).map(|i| Some(i as i64)));
            values.extend(std::iter::repeat_n(None, n_nulls));

            let col: ArrayRef = Arc::new(Int64Array::from(values));
            let null_f = evaluate(&col, &ScalarPredicate::IsNull).unwrap();
            let not_null_f = evaluate(&col, &ScalarPredicate::IsNotNull).unwrap();

            let union = null_f.union(&not_null_f);
            prop_assert_eq!(union.len() as usize, total);
        }
    }
}

// ─── Persist tests ────────────────────────────────────────────────────────────
#[cfg(feature = "persist")]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod persist_tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch};
    use arrow_row::SortField;
    use arrow_schema::{DataType, Field, Schema};

    use crate::filter::FilterIndex;
    use crate::permutation::PermutationIndex;
    use crate::persist::{load_filter, load_permutation, save_filter, save_permutation};
    use crate::sort_builder::SortBuilder;

    fn build_test_index(values: &[i64]) -> PermutationIndex {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))])
            .expect("batch");
        let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("builder");
        builder.push(&batch, &[0]).expect("push");
        builder.finish().expect("finish")
    }

    #[test]
    fn persist_permutation_round_trip() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("perm.avs");

        let original = build_test_index(&[30, 10, 20, 50, 40]);
        let orig_ids = original.read_range(0, original.len() as usize);

        save_permutation(&original, &path).expect("save");
        let loaded = load_permutation(&path).expect("load");

        assert_eq!(loaded.len(), original.len());
        let loaded_ids = loaded.read_range(0, loaded.len() as usize);
        assert_eq!(loaded_ids, orig_ids);
    }

    #[test]
    fn persist_permutation_loads_as_mmap_persisted() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("perm.avs");

        let original = build_test_index(&[3, 1, 2]);
        save_permutation(&original, &path).expect("save");
        let loaded = load_permutation(&path).expect("load");

        let debug = format!("{loaded:?}");
        assert!(debug.contains("MmapPersisted"), "got: {debug}");
    }

    #[test]
    fn persist_permutation_identity() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("nat.avs");

        let natural = PermutationIndex::natural(1000).expect("natural");
        save_permutation(&natural, &path).expect("save");
        let loaded = load_permutation(&path).expect("load");

        let ids = loaded.read_range(0, 1000);
        let expected: Vec<u32> = (0..1000).collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn persist_filter_round_trip() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("filt.avs");

        let original = FilterIndex::from_ids([1, 3, 5, 7, 999]);
        save_filter(&original, &path).expect("save");
        let loaded = load_filter(&path).expect("load");

        assert_eq!(loaded.len(), 5);
        for id in [1, 3, 5, 7, 999] {
            assert!(loaded.contains(id), "missing {id}");
        }
        assert!(!loaded.contains(0));
        assert!(!loaded.contains(2));
    }

    #[test]
    fn persist_filter_empty() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("empty.avs");

        let original = FilterIndex::from_ids(std::iter::empty::<u32>());
        save_filter(&original, &path).expect("save");
        let loaded = load_filter(&path).expect("load");

        assert!(loaded.is_empty());
    }

    #[test]
    fn persist_filter_full_range() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("full.avs");

        let original = FilterIndex::from_ids(0..10_000);
        save_filter(&original, &path).expect("save");
        let loaded = load_filter(&path).expect("load");

        assert_eq!(loaded.len(), 10_000);
        assert!(loaded.contains(0));
        assert!(loaded.contains(9_999));
    }

    #[test]
    fn persist_invalid_magic() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("garbage.avs");

        std::fs::write(&path, b"GARBAGE!0000000000000000").expect("write");
        let err = load_permutation(&path).unwrap_err();
        assert!(err.to_string().contains("invalid magic"), "got: {err}");

        let err = load_filter(&path).unwrap_err();
        assert!(err.to_string().contains("invalid magic"), "got: {err}");
    }

    #[test]
    fn persist_truncated_file() {
        let dir = tempfile::tempdir().expect("tmpdir");

        // Too small for header
        let path = dir.path().join("tiny.avs");
        std::fs::write(&path, b"short").expect("write");
        let err = load_permutation(&path).unwrap_err();
        assert!(err.to_string().contains("too small"), "got: {err}");

        // Valid header but claims more rows than present
        let path2 = dir.path().join("trunc.avs");
        let mut data = Vec::new();
        data.extend_from_slice(b"AVSPI\x00\x02\x00");
        data.extend_from_slice(&1000u64.to_le_bytes()); // claims 1000 rows
        // but no row data follows
        std::fs::write(&path2, &data).expect("write");
        let err = load_permutation(&path2).unwrap_err();
        assert!(err.to_string().contains("truncated"), "got: {err}");
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn persist_permutation_large() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("large.avs");

        let n: u64 = 1_000_000;
        let natural = PermutationIndex::natural(n).expect("natural");
        save_permutation(&natural, &path).expect("save");
        let loaded = load_permutation(&path).expect("load");

        assert_eq!(loaded.len(), n);

        // Spot-check ranges
        let first_100 = loaded.read_range(0, 100);
        let expected: Vec<u32> = (0u32..100).collect();
        assert_eq!(first_100, expected);

        let n32 = n as u32;
        let last_100 = loaded.read_range(n as usize - 100, n as usize);
        let expected: Vec<u32> = (n32 - 100..n32).collect();
        assert_eq!(last_100, expected);
    }
}

#[cfg(feature = "persist")]
#[allow(clippy::cast_possible_truncation)]
mod persist_proptests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch};
    use arrow_row::SortField;
    use arrow_schema::{DataType, Field, Schema};
    use proptest::prelude::*;

    use crate::filter::FilterIndex;
    use crate::persist::{load_filter, load_permutation, save_filter, save_permutation};
    use crate::sort_builder::SortBuilder;

    proptest! {
        #[test]
        fn prop_persist_permutation_identity(
            values in prop::collection::vec(-1000i64..1000, 1..500),
        ) {
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
            let batch = RecordBatch::try_new(
                schema,
                vec![Arc::new(Int64Array::from(values.clone()))],
            ).expect("batch");

            let mut builder = SortBuilder::new(vec![SortField::new(DataType::Int64)]).expect("builder");
            builder.push(&batch, &[0]).expect("push");
            let original = builder.finish().expect("finish");

            let dir = tempfile::tempdir().expect("tmpdir");
            let path = dir.path().join("prop_perm.avs");

            save_permutation(&original, &path).expect("save");
            let loaded = load_permutation(&path).expect("load");

            let orig_ids = original.read_range(0, original.len() as usize);
            let loaded_ids = loaded.read_range(0, loaded.len() as usize);
            prop_assert_eq!(orig_ids, loaded_ids);
        }

        #[test]
        fn prop_persist_filter_identity(
            ids in prop::collection::vec(0u32..100_000, 0..500),
        ) {
            let original = FilterIndex::from_ids(ids.iter().copied());

            let dir = tempfile::tempdir().expect("tmpdir");
            let path = dir.path().join("prop_filt.avs");

            save_filter(&original, &path).expect("save");
            let loaded = load_filter(&path).expect("load");

            prop_assert_eq!(original.len(), loaded.len());
            for &id in &ids {
                prop_assert!(loaded.contains(id), "missing {}", id);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HashIndex tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "hash-index")]
mod hash_index_tests {
    use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    use crate::hash_index::HashIndex;
    use crate::scalar_value::OwnedScalar;

    #[test]
    fn hash_index_int64_lookup() {
        let array = Int64Array::from(vec![10, 20, 30, 20, 10]);
        let idx = HashIndex::build(&array).expect("build");
        let result = idx.lookup(&OwnedScalar::Int64(20));
        assert_eq!(result.len(), 2);
        assert!(result.contains(1));
        assert!(result.contains(3));
    }

    #[test]
    fn hash_index_string_lookup() {
        let array = StringArray::from(vec!["a", "b", "a", "c"]);
        let idx = HashIndex::build(&array).expect("build");
        let result = idx.lookup(&OwnedScalar::Utf8("a".to_owned()));
        assert_eq!(result.len(), 2);
        assert!(result.contains(0));
        assert!(result.contains(2));
    }

    #[test]
    fn hash_index_null_handling() {
        let array = Int64Array::from(vec![Some(1), None, Some(1), None]);
        let idx = HashIndex::build(&array).expect("build");
        let result = idx.lookup(&OwnedScalar::Null);
        assert_eq!(result.len(), 2);
        assert!(result.contains(1));
        assert!(result.contains(3));
    }

    #[test]
    fn hash_index_lookup_missing() {
        let array = Int64Array::from(vec![1, 2, 3]);
        let idx = HashIndex::build(&array).expect("build");
        let result = idx.lookup(&OwnedScalar::Int64(99));
        assert!(result.is_empty());
    }

    #[test]
    fn hash_index_lookup_many() {
        let array = Int64Array::from(vec![10, 20, 30, 20, 10]);
        let idx = HashIndex::build(&array).expect("build");
        let result = idx.lookup_many(&[OwnedScalar::Int64(10), OwnedScalar::Int64(30)]);
        assert_eq!(result.len(), 3);
        assert!(result.contains(0));
        assert!(result.contains(2));
        assert!(result.contains(4));
    }

    #[test]
    fn hash_index_distinct_count() {
        let array = Int64Array::from(vec![1, 2, 2, 3, 3, 3]);
        let idx = HashIndex::build(&array).expect("build");
        assert_eq!(idx.distinct_count(), 3);
    }

    #[test]
    fn hash_index_value_counts() {
        let array = Int64Array::from(vec![1, 2, 2, 3, 3, 3]);
        let idx = HashIndex::build(&array).expect("build");
        let counts: std::collections::HashMap<_, _> = idx.value_counts().collect();
        assert_eq!(counts[&OwnedScalar::Int64(1)], 1);
        assert_eq!(counts[&OwnedScalar::Int64(2)], 2);
        assert_eq!(counts[&OwnedScalar::Int64(3)], 3);
    }

    #[test]
    fn hash_index_float64_nan() {
        let array = Float64Array::from(vec![1.0, 2.0, f64::NAN, f64::NAN]);
        let idx = HashIndex::build(&array).expect("build");
        let result = idx.lookup(&OwnedScalar::Float64(ordered_float::OrderedFloat(f64::NAN)));
        assert_eq!(result.len(), 2);
        assert!(result.contains(2));
        assert!(result.contains(3));
    }

    #[test]
    fn hash_index_build_batches() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let b1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![10, 20]))],
        )
        .expect("batch1");
        let b2 = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![20, 30]))])
            .expect("batch2");

        let idx = HashIndex::build_batches(vec![(b1, 0), (b2, 0)].into_iter()).expect("build");
        assert_eq!(idx.total_rows(), 4);

        let result = idx.lookup(&OwnedScalar::Int64(20));
        assert_eq!(result.len(), 2);
        assert!(result.contains(1)); // batch1 row1
        assert!(result.contains(2)); // batch2 row0
    }

    #[test]
    fn hash_index_empty_array() {
        let array = Int64Array::from(Vec::<i64>::new());
        let idx = HashIndex::build(&array).expect("build");
        assert_eq!(idx.distinct_count(), 0);
        assert_eq!(idx.total_rows(), 0);
        assert!(idx.lookup(&OwnedScalar::Int64(1)).is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// InvertedIndex tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "inverted-index")]
mod inverted_index_tests {
    use arrow_array::StringArray;

    use crate::inverted_index::{InvertedIndex, NgramTokenizer, WhitespaceTokenizer};

    #[test]
    fn inverted_search_single_token() {
        let array = StringArray::from(vec!["hello world", "hello", "world"]);
        let tok = WhitespaceTokenizer;
        let idx = InvertedIndex::build(&array, &tok).expect("build");
        let result = idx.search("hello", &tok);
        assert_eq!(result.len(), 2);
        assert!(result.contains(0));
        assert!(result.contains(1));
    }

    #[test]
    fn inverted_search_multi_token_and() {
        let array = StringArray::from(vec!["hello world", "hello", "world"]);
        let tok = WhitespaceTokenizer;
        let idx = InvertedIndex::build(&array, &tok).expect("build");
        let result = idx.search("hello world", &tok);
        assert_eq!(result.len(), 1);
        assert!(result.contains(0));
    }

    #[test]
    fn inverted_search_not_found() {
        let array = StringArray::from(vec!["hello world", "foo"]);
        let tok = WhitespaceTokenizer;
        let idx = InvertedIndex::build(&array, &tok).expect("build");
        let result = idx.search("xyz", &tok);
        assert!(result.is_empty());
    }

    #[test]
    fn inverted_search_prefix() {
        let array = StringArray::from(vec!["apple", "application", "banana"]);
        let tok = WhitespaceTokenizer;
        let idx = InvertedIndex::build(&array, &tok).expect("build");
        let result = idx.search_prefix("app");
        assert_eq!(result.len(), 2);
        assert!(result.contains(0));
        assert!(result.contains(1));
    }

    #[test]
    fn inverted_case_insensitive() {
        let array = StringArray::from(vec!["Hello"]);
        let tok = WhitespaceTokenizer;
        let idx = InvertedIndex::build(&array, &tok).expect("build");
        let result = idx.search("hello", &tok);
        assert_eq!(result.len(), 1);
        assert!(result.contains(0));
    }

    #[test]
    fn inverted_null_handling() {
        let array = StringArray::from(vec![Some("hello"), None, Some("world")]);
        let tok = WhitespaceTokenizer;
        let idx = InvertedIndex::build(&array, &tok).expect("build");
        assert_eq!(idx.total_rows(), 3);
        let result = idx.search("hello", &tok);
        assert_eq!(result.len(), 1);
        assert!(result.contains(0));
    }

    #[test]
    fn inverted_ngram_tokenizer() {
        let tok = NgramTokenizer { n: 3 };
        let tokens = crate::inverted_index::Tokenizer::tokenize(&tok, "hello");
        assert_eq!(tokens, vec!["hel", "ell", "llo"]);
    }

    #[test]
    fn inverted_whitespace_tokenizer() {
        let tok = WhitespaceTokenizer;
        let tokens = crate::inverted_index::Tokenizer::tokenize(&tok, "foo  bar  baz");
        assert_eq!(tokens, vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn inverted_token_count() {
        let array = StringArray::from(vec!["hello world", "hello"]);
        let tok = WhitespaceTokenizer;
        let idx = InvertedIndex::build(&array, &tok).expect("build");
        assert_eq!(idx.token_count(), 2); // "hello", "world"
    }

    #[test]
    fn inverted_empty_query() {
        let array = StringArray::from(vec!["hello"]);
        let tok = WhitespaceTokenizer;
        let idx = InvertedIndex::build(&array, &tok).expect("build");
        let result = idx.search("", &tok);
        assert!(result.is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HashIndex property tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "hash-index")]
#[allow(clippy::cast_possible_truncation)]
mod hash_index_proptests {
    use arrow_array::Int64Array;
    use proptest::prelude::*;

    use crate::hash_index::HashIndex;
    use crate::scalar_value::OwnedScalar;

    proptest! {
        #[test]
        fn prop_hash_index_lookup_matches_scan(
            values in prop::collection::vec(-100i64..100, 1..200),
            needle in -100i64..100,
        ) {
            let array = Int64Array::from(values.clone());
            let idx = HashIndex::build(&array).expect("build");
            let result = idx.lookup(&OwnedScalar::Int64(needle));

            // Manual scan
            let expected: Vec<u32> = values
                .iter()
                .enumerate()
                .filter(|(_, v)| **v == needle)
                .map(|(i, _)| i as u32)
                .collect();

            prop_assert_eq!(result.len(), expected.len() as u64);
            for id in expected {
                prop_assert!(result.contains(id));
            }
        }

        #[test]
        fn prop_hash_index_all_values_covered(
            values in prop::collection::vec(-50i64..50, 1..200),
        ) {
            let array = Int64Array::from(values.clone());
            let idx = HashIndex::build(&array).expect("build");

            for (i, &v) in values.iter().enumerate() {
                let result = idx.lookup(&OwnedScalar::Int64(v));
                prop_assert!(result.contains(i as u32), "row {} with value {} not found", i, v);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// InvertedIndex property tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "inverted-index")]
#[allow(clippy::redundant_closure_for_method_calls)]
mod inverted_index_proptests {
    use arrow_array::StringArray;
    use proptest::prelude::*;

    use crate::inverted_index::{InvertedIndex, WhitespaceTokenizer};

    proptest! {
        #[test]
        fn prop_inverted_search_is_subset(
            texts in prop::collection::vec("[a-z ]{1,20}", 1..50),
            query in "[a-z]{1,10}",
        ) {
            let array = StringArray::from(texts.iter().map(|s| s.as_str()).collect::<Vec<_>>());
            let tok = WhitespaceTokenizer;
            let idx = InvertedIndex::build(&array, &tok).expect("build");
            let result = idx.search(&query, &tok);

            // Every matched row id must be in range
            for id in result.iter() {
                prop_assert!((id as usize) < texts.len());
            }
        }
    }
}
