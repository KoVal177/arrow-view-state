//! Tests for the arrow-view-state permutation index engine.

#[cfg(test)]
mod unit {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray};
    use arrow_schema::SortOptions;

    use crate::builder::{build, refine};
    use crate::filter::FilterIndex;
    use crate::permutation::PermutationIndex;

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
        idx.indices().values().to_vec()
    }

    // ── Natural (identity) permutation ─────────────────────────────

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

    // ── Build: single column sorts ─────────────────────────────────

    #[test]
    fn sort_int64_ascending() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![30, 10, 20]));
        let idx = build(&[col], &[asc()]).expect("build");
        assert_eq!(ids(&idx), vec![1, 2, 0]); // 10, 20, 30
    }

    #[test]
    fn sort_int64_descending() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![30, 10, 20]));
        let idx = build(&[col], &[desc()]).expect("build");
        assert_eq!(ids(&idx), vec![0, 2, 1]); // 30, 20, 10
    }

    #[test]
    fn sort_int32_with_nulls_asc() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![Some(30), None, Some(10), None]));
        let idx = build(&[col], &[asc()]).expect("build");
        let result = ids(&idx);
        // ascending, nulls_first=false → non-nulls sorted first, then nulls
        assert_eq!(result[0], 2); // 10
        assert_eq!(result[1], 0); // 30
        // nulls at end
    }

    #[test]
    fn sort_strings_ascending() {
        let col: ArrayRef = Arc::new(StringArray::from(vec!["cherry", "apple", "banana"]));
        let idx = build(&[col], &[asc()]).expect("build");
        assert_eq!(ids(&idx), vec![1, 2, 0]); // apple, banana, cherry
    }

    #[test]
    fn sort_float64_ascending() {
        #[allow(clippy::approx_constant)]
        let col: ArrayRef = Arc::new(Float64Array::from(vec![3.14, 1.0, 2.71]));
        let idx = build(&[col], &[asc()]).expect("build");
        assert_eq!(ids(&idx), vec![1, 2, 0]); // 1.0, 2.71, 3.14
    }

    #[test]
    fn sort_empty_column() {
        let col: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        let idx = build(&[col], &[asc()]).expect("build");
        assert!(idx.is_empty());
    }

    // ── Build: multi-column sort ───────────────────────────────────

    #[test]
    fn multi_column_sort() {
        // Sort by category (asc), then by value (desc)
        let cat: ArrayRef = Arc::new(StringArray::from(vec!["B", "A", "B", "A", "C"]));
        let val: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
        let idx = build(&[cat, val], &[asc(), desc()]).expect("build");
        let result = ids(&idx);
        // A(40, row3), A(20, row1), B(30, row2), B(10, row0), C(50, row4)
        assert_eq!(result, vec![3, 1, 2, 0, 4]);
    }

    // ── Build: stability for equal values ──────────────────────────

    #[test]
    fn equal_values_produces_valid_permutation() {
        let col: ArrayRef = Arc::new(StringArray::from(vec![
            "same", "same", "same", "same", "same",
        ]));
        let idx = build(&[col], &[asc()]).expect("build");
        // For equal values, unstable sort may reorder — we only guarantee
        // it's a valid permutation.
        let mut result = ids(&idx);
        result.sort_unstable();
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    // ── Refine: stable tiebreaker ──────────────────────────────────

    #[test]
    fn refine_preserves_prior_for_ties() {
        let primary: ArrayRef = Arc::new(Int64Array::from(vec![30, 10, 20, 40, 50]));
        let tiebreaker: ArrayRef = Arc::new(StringArray::from(vec!["B", "A", "A", "B", "A"]));

        let sorted = build(&[primary], &[asc()]).expect("build");
        // Sorted by primary: [1(10), 2(20), 0(30), 3(40), 4(50)]
        assert_eq!(ids(&sorted), vec![1, 2, 0, 3, 4]);

        let refined = refine(sorted, &[tiebreaker], &[asc()]).expect("refine");
        let result = ids(&refined);
        // Stable sort by tiebreaker: A(1), A(2), A(4), B(0), B(3)
        assert_eq!(result, vec![1, 2, 4, 0, 3]);
    }

    #[test]
    fn refine_length_mismatch() {
        let col1: ArrayRef = Arc::new(Int64Array::from(vec![3, 1, 2]));
        let col2: ArrayRef = Arc::new(Int64Array::from(vec![10, 20])); // wrong length
        let sorted = build(&[col1], &[asc()]).expect("build");
        let result = refine(sorted, &[col2], &[asc()]);
        assert!(result.is_err());
        assert!(
            result
                .expect_err("should be LengthMismatch")
                .to_string()
                .contains("length mismatch")
        );
    }

    // ── Slice ──────────────────────────────────────────────────────

    #[test]
    fn slice_subrange() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![50, 10, 30, 20, 40]));
        let idx = build(&[col], &[asc()]).expect("build");
        // Sorted: 10(1), 20(3), 30(2), 40(4), 50(0)
        let sliced = idx.slice(2, 2);
        assert_eq!(ids(&sliced), vec![2, 4]); // 30(2), 40(4)
    }

    // ── Apply ──────────────────────────────────────────────────────

    #[test]
    fn apply_reorders_array() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![30, 10, 20]));
        let idx = build(&[col.clone()], &[asc()]).expect("build");
        let reordered = idx.apply(col.as_ref()).expect("apply");
        let values: Vec<i64> = reordered
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array")
            .values()
            .to_vec();
        assert_eq!(values, vec![10, 20, 30]);
    }

    // ── Error cases ────────────────────────────────────────────────

    #[test]
    fn empty_columns_error() {
        let result = build(&[], &[]);
        assert!(result.is_err());
        assert!(
            result
                .expect_err("should be EmptyColumns")
                .to_string()
                .contains("empty")
        );
    }

    #[test]
    fn too_many_rows_error() {
        let result = PermutationIndex::natural(u64::from(u32::MAX) + 1);
        assert!(result.is_err());
    }

    #[test]
    fn column_length_mismatch_error() {
        let col1: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let col2: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        let result = build(&[col1, col2], &[asc(), asc()]);
        assert!(result.is_err());
        assert!(
            result
                .expect_err("should be LengthMismatch")
                .to_string()
                .contains("length mismatch")
        );
    }

    // ── FilterIndex ────────────────────────────────────────────────

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
        assert!(!filter.contains(1)); // false
        assert!(!filter.contains(2)); // null → false
        assert!(filter.contains(3));
    }

    #[test]
    fn filter_apply_to_permutation() {
        let col: ArrayRef = Arc::new(Int64Array::from(vec![50, 10, 30, 20, 40]));
        let sorted = build(&[col], &[asc()]).expect("build");
        // Sorted: [1(10), 3(20), 2(30), 4(40), 0(50)]

        // Filter: keep only rows 0, 2, 4 (values 50, 30, 40)
        let filter = FilterIndex::from_ids([0, 2, 4]);
        let filtered = filter.apply_to_permutation(&sorted);
        // In sort order, filtered: 2(30), 4(40), 0(50)
        assert_eq!(ids(&filtered), vec![2, 4, 0]);
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
}

#[cfg(test)]
mod proptests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array};
    use arrow_schema::SortOptions;
    use proptest::prelude::*;

    use crate::builder::build;
    use crate::filter::FilterIndex;
    use crate::permutation::PermutationIndex;

    fn asc() -> SortOptions {
        SortOptions {
            descending: false,
            nulls_first: false,
        }
    }

    proptest! {
        #[test]
        fn natural_always_in_bounds(n in 1u64..100_000) {
            let idx = PermutationIndex::natural(n).expect("natural");
            for &id in idx.indices().values() {
                prop_assert!(u64::from(id) < n);
            }
        }

        #[test]
        fn sorted_index_is_permutation(
            values in prop::collection::vec(-1000i64..1000, 1..500),
        ) {
            let n = values.len();
            let col: ArrayRef = Arc::new(Int64Array::from(values));
            let idx = build(&[col], &[asc()]).expect("build");

            // Every physical row ID appears exactly once.
            let mut sorted_ids: Vec<u32> = idx.indices().values().to_vec();
            sorted_ids.sort_unstable();
            #[allow(clippy::cast_possible_truncation)]
            let expected: Vec<u32> = (0..n as u32).collect();
            prop_assert_eq!(sorted_ids, expected, "index must be a permutation");
        }

        #[test]
        fn sorted_values_are_monotonic(
            values in prop::collection::vec(-1000i64..1000, 1..500),
        ) {
            let n = values.len();
            let col: ArrayRef = Arc::new(Int64Array::from(values.clone()));
            let idx = build(&[col], &[asc()]).expect("build");
            let perm = idx.indices().values();

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
        fn filter_preserves_sort_order(
            values in prop::collection::vec(-1000i64..1000, 2..200),
            keep_fraction in 0.1f64..0.9,
        ) {
            let n = values.len();
            let col: ArrayRef = Arc::new(Int64Array::from(values.clone()));
            let sorted = build(&[col], &[asc()]).expect("build");

            // Build a filter that keeps approximately keep_fraction of rows
            #[allow(clippy::cast_possible_truncation)]
            let keep_ids: Vec<u32> = (0..n as u32)
                .filter(|i| (*i as f64 / n as f64) < keep_fraction)
                .collect();

            if keep_ids.is_empty() {
                return Ok(());
            }

            let filter = FilterIndex::from_ids(keep_ids);
            let filtered = filter.apply_to_permutation(&sorted);

            // Filtered result should still be in sorted order
            let perm = filtered.indices().values();
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
    }
}
